#include <array>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <clocale>
#include <cstring>
#include <random>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <thread>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <exception>
#include <ctime>
#include <mutex>
#include <memory>
#include <deque>
#include <torch/torch.h>
#include <ATen/autocast_mode.h>
#include <ATen/cuda/CUDAContext.h>
#if defined(_MSC_VER)
#include <intrin.h>
#if defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#endif
#else
#if defined(__x86_64__) || defined(__i386)
#include <immintrin.h>
#include <cpuid.h>
#endif
#endif

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

using namespace std;
using namespace chrono;
static void clearConsoleFull() {
#if defined(_WIN32)
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut == INVALID_HANDLE_VALUE) return;

    CONSOLE_SCREEN_BUFFER_INFO csbi;
    if (!GetConsoleScreenBufferInfo(hOut, &csbi)) return;

    const DWORD cellCount = (DWORD)csbi.dwSize.X * (DWORD)csbi.dwSize.Y;
    const COORD home{ 0, 0 };
    DWORD written = 0;

    FillConsoleOutputCharacterA(hOut, ' ', cellCount, home, &written);
    FillConsoleOutputAttribute(hOut, csbi.wAttributes, cellCount, home, &written);
    SetConsoleCursorPosition(hOut, home);
#else
    std::cout << "\033[2J\033[H";
#endif
}


#if defined(_MSC_VER)
#define AI_FORCEINLINE __forceinline
#define AI_HOT
#define AI_LIKELY(x)   (x)
#define AI_UNLIKELY(x) (x)
#else
#if defined(__GNUC__)
#define AI_FORCEINLINE inline __attribute__((always_inline))
#define AI_HOT __attribute__((hot))
#define AI_LIKELY(x)   (__builtin_expect(!!(x), 1))
#define AI_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#else
#define AI_FORCEINLINE inline
#define AI_HOT
#define AI_LIKELY(x)   (x)
#define AI_UNLIKELY(x) (x)
#endif
#endif

static AI_FORCEINLINE int ctz64(uint64_t x) {
#if defined(_MSC_VER) && defined(_M_X64)
    unsigned long idx = 0;
    _BitScanForward64(&idx, x);
    return (int)idx;
#elif defined(_MSC_VER)
    unsigned long idx = 0;
    uint32_t lo = (uint32_t)x;
    if (lo) { _BitScanForward(&idx, lo); return (int)idx; }
    uint32_t hi = (uint32_t)(x >> 32);
    _BitScanForward(&idx, hi);
    return (int)idx + 32;
#else
    return __builtin_ctzll(x);
#endif
}

static AI_FORCEINLINE int clz64(uint64_t x) {
#if defined(_MSC_VER) && defined(_M_X64)
    unsigned long idx = 0;
    _BitScanReverse64(&idx, x);
    return 63 - (int)idx;
#elif defined(_MSC_VER)
    unsigned long idx = 0;
    uint32_t hi = (uint32_t)(x >> 32);
    if (hi) { _BitScanReverse(&idx, hi); return 31 - (int)idx; }
    uint32_t lo = (uint32_t)x;
    _BitScanReverse(&idx, lo);
    return 63 - ((int)idx + 32);
#else
    return __builtin_clzll(x);
#endif
}

static AI_FORCEINLINE int popcount64(uint64_t x) {
#if defined(_MSC_VER) && defined(_M_X64)
    return (int)__popcnt64(x);
#elif defined(_MSC_VER)
    return (int)(__popcnt((uint32_t)x) + __popcnt((uint32_t)(x >> 32)));
#else
    return __builtin_popcountll(x);
#endif
}

static AI_FORCEINLINE int iabs(int x) { return x < 0 ? -x : x; }

// ----- CPUID helpers -----
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
static AI_FORCEINLINE void cpuid_ex(int leaf, int subleaf, int& a, int& b, int& c, int& d) {
    int r[4];
    __cpuidex(r, leaf, subleaf);
    a = r[0]; b = r[1]; c = r[2]; d = r[3];
}
static AI_FORCEINLINE bool cpuHasBMI2() {
    int a, b, c, d;
    cpuid_ex(7, 0, a, b, c, d);
    return (b & (1 << 8)) != 0;
}
static AI_FORCEINLINE void cpuVendorFamilyModel(string& vendor, int& family, int& model) {
    int a, b, c, d;
    cpuid_ex(0, 0, a, b, c, d);
    char v[13];
    memcpy(v + 0, &b, 4);
    memcpy(v + 4, &d, 4);
    memcpy(v + 8, &c, 4);
    v[12] = 0;
    vendor = v;

    cpuid_ex(1, 0, a, b, c, d);
    int baseFamily = (a >> 8) & 0xF;
    int baseModel = (a >> 4) & 0xF;
    int extFamily = (a >> 20) & 0xFF;
    int extModel = (a >> 16) & 0xF;

    family = baseFamily;
    if (baseFamily == 0xF) family += extFamily;

    model = baseModel;
    if (baseFamily == 0x6 || baseFamily == 0xF) model |= (extModel << 4);
}
static AI_FORCEINLINE bool shouldUsePextPolicy() {
    if (!cpuHasBMI2()) return false;

    string vendor;
    int fam = 0, mod = 0;
    cpuVendorFamilyModel(vendor, fam, mod);

    if (vendor == "GenuineIntel") return (fam == 6 && mod >= 0x3C);
    if (vendor == "AuthenticAMD") return (fam >= 0x19);
    return false;
}
#else
#if defined(__x86_64__) || defined(__i386)
static AI_FORCEINLINE void cpuid(uint32_t leaf, uint32_t subleaf,
    uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
    __cpuid_count(leaf, subleaf, a, b, c, d);
}
static bool cpuHasBMI2() {
    uint32_t a, b, c, d;
    cpuid(7, 0, a, b, c, d);
    return (b & (1u << 8)) != 0;
}
static void cpuVendorFamilyModel(string& vendor, int& family, int& model) {
    uint32_t a, b, c, d;
    cpuid(0, 0, a, b, c, d);
    char v[13];
    memcpy(v + 0, &b, 4);
    memcpy(v + 4, &d, 4);
    memcpy(v + 8, &c, 4);
    v[12] = 0;
    vendor = v;

    cpuid(1, 0, a, b, c, d);
    int baseFamily = (a >> 8) & 0xF;
    int baseModel = (a >> 4) & 0xF;
    int extFamily = (a >> 20) & 0xFF;
    int extModel = (a >> 16) & 0xF;

    family = baseFamily;
    if (baseFamily == 0xF) family += extFamily;

    model = baseModel;
    if (baseFamily == 0x6 || baseFamily == 0xF) model |= (extModel << 4);
}
static bool shouldUsePextPolicy() {
    if (!cpuHasBMI2()) return false;

    string vendor;
    int fam = 0, mod = 0;
    cpuVendorFamilyModel(vendor, fam, mod);

    if (vendor == "GenuineIntel") return (fam == 6 && mod >= 0x3C);
    if (vendor == "AuthenticAMD") return (fam >= 0x19);
    return false;
}
#else
static bool shouldUsePextPolicy() { return false; }
#endif
#endif

#ifndef AI_ENABLE_PEXT
#define AI_ENABLE_PEXT 1   // or 0, whichever default you intended
#endif

#if defined(_MSC_VER) && AI_ENABLE_PEXT && (defined(_M_X64) || defined(_M_IX86))
static AI_FORCEINLINE uint64_t pext_u64_runtime(uint64_t x, uint64_t m) { return _pext_u64(x, m); }
#define HAVE_PEXT_INTRIN 1
#elif (defined(__GNUC__) && (defined(__x86_64__) || defined(__i386)))
__attribute__((target("bmi2")))
static inline uint64_t pext_u64_runtime(uint64_t x, uint64_t m) { return _pext_u64(x, m); }
#define HAVE_PEXT_INTRIN 1
#else
#define HAVE_PEXT_INTRIN 0
#endif



thread_local std::mt19937 Random(std::random_device{}());
thread_local std::uniform_int_distribution<int> Range(0, 215);

static inline int randInt(int n) {
    std::uniform_int_distribution<int> d(0, n - 1);
    return d(Random);
}



static AI_FORCEINLINE int pop_lsb(uint64_t& bb) {
    int sq = ctz64(bb);
    bb &= (bb - 1);
    return sq;
}



struct Position {
    array<uint64_t, 2> color;
    array<uint64_t, 6> piece;
    int side;
    array<uint64_t, 2> ep1;
    uint64_t ep2;
    array<int, 4> rook;
    int castle;
    int dice;
    uint64_t key;
};

struct MoveList {
    int n;
    int m[255];
};


struct moveState {
    int   move;
    float eval;
    uint32_t visits;
    float prior;
    uint64_t pvKey;
    std::vector<int> pv;
    double dif;
};


static AI_FORCEINLINE void atomicAddFloat(std::atomic<float>& a, float add) {
    float old = a.load(std::memory_order_relaxed);
    while (!a.compare_exchange_weak(old, old + add,
        std::memory_order_release,
        std::memory_order_relaxed)) {
        // old updated on failure
    }
}
static AI_FORCEINLINE void atomicAddDouble(std::atomic<double>& a, double add) {
    double old = a.load(std::memory_order_relaxed);
    while (!a.compare_exchange_weak(old, old + add,
        std::memory_order_release,
        std::memory_order_relaxed)) {
        // old updated on failure
    }
}
struct TTEdge {
    std::atomic<double> valueSum{ 0.0 };
    std::atomic<uint32_t> visits{ 0 };
    uint16_t move = 0;
    uint16_t priorQ = 0;

    AI_FORCEINLINE float prior() const {
        return (float)priorQ * (1.0f / 65535.0f);
    }

    AI_FORCEINLINE void setPrior(float p) {
        if (!(p > 0.0f)) p = 0.0f;
        else if (p > 1.0f) p = 1.0f;
        priorQ = (uint16_t)lrintf(p * 65535.0f);
    }

    AI_FORCEINLINE uint16_t priorRaw() const {
        return priorQ;
    }

    AI_FORCEINLINE void setPriorRaw(uint16_t q) {
        priorQ = q;
    }

    AI_FORCEINLINE double sum() const {
        return valueSum.load(std::memory_order_relaxed);
    }

    AI_FORCEINLINE void addVisitAndValue(float v) {
        atomicAddDouble(valueSum, (double)v);
        visits.fetch_add(1, std::memory_order_release);
    }
};

struct TTNode {
    uint64_t key = 0;
    uint32_t edgeBegin = 0;
    uint8_t  edgeCount = 0;
    std::atomic<uint8_t> expanded{ 0 };

    uint8_t terminal = 0;
    uint8_t chance = 0;

    std::atomic<double> valueSum{ 0.0f };
    std::atomic<uint32_t> visits{ 0 };
    std::atomic<uint32_t> chanceCursor{ 0 };

    AI_FORCEINLINE bool isExpanded() const {
        return expanded.load(std::memory_order_acquire) != 0;
    }

    AI_FORCEINLINE double sum() const {
        return valueSum.load(std::memory_order_relaxed);
    }
    AI_FORCEINLINE void addVisitAndValue(float v) {
        atomicAddDouble(valueSum, (double)v);
        visits.fetch_add(1, std::memory_order_release);
    }
    AI_FORCEINLINE void publish(uint64_t k, uint32_t begin, uint8_t count,
        int term, int isChance) {
        key = k;
        edgeBegin = begin;
        edgeCount = count;
        terminal = (uint8_t)term;
        chance = (uint8_t)isChance;
        expanded.store(1, std::memory_order_release);
    }
};



static constexpr uint64_t FILE_A = 0x0101010101010101ULL;
static constexpr uint64_t FILE_H = 0x8080808080808080ULL;
static constexpr uint64_t RANK_1 = 0x00000000000000FFULL;
static constexpr uint64_t RANK_2 = 0x000000000000FF00ULL;
static constexpr uint64_t RANK_3 = 0x0000000000FF0000ULL;
static constexpr uint64_t RANK_4 = 0x00000000FF000000ULL;
static constexpr uint64_t RANK_5 = 0x000000FF00000000ULL;
static constexpr uint64_t RANK_6 = 0x0000FF0000000000ULL;
static constexpr uint64_t RANK_7 = 0x00FF000000000000ULL;
static constexpr uint64_t RANK_8 = 0xFF00000000000000ULL;



static AI_FORCEINLINE int fileIndex(char f) { return (f >= 'a' && f <= 'h') ? (f - 'a') : -1; }
static AI_FORCEINLINE int rankIndex(char r) { return (r >= '1' && r <= '8') ? (r - '1') : -1; }

static AI_FORCEINLINE int sqFromName2(char f, char r) {
    int fi = fileIndex((char)tolower((unsigned char)f));
    int ri = rankIndex(r);
    if (fi < 0 || ri < 0) return -1;
    return ri * 8 + fi; // a1=0 ... h8=63
}

static AI_FORCEINLINE string sqName(int sq) {
    if (sq < 0 || sq >= 64) return "-";
    char f = char('a' + (sq & 7));
    char r = char('1' + (sq >> 3));
    string s; s += f; s += r;
    return s;
}

static AI_FORCEINLINE uint64_t bit(int sq) { return 1ULL << sq; }

static AI_FORCEINLINE int Piece(const Position& pos, int sq) {
    const uint64_t b = bit(sq);
    for (int pt = 0; pt < 6; ++pt) if (pos.piece[pt] & b) return pt;
    return -1;
}

static string bbToSquares(uint64_t bb) {
    if (!bb) return "-";
    string out;
    bool first = true;
    while (bb) {
        int sq = pop_lsb(bb);
        if (!first) out += ' ';
        out += sqName(sq);
        first = false;
    }
    return out;
}
static vector<string> g_diceTable;
static unordered_map<string, int> g_diceIndex;
static array<uint8_t, 84> g_diceMask;

alignas(64) static array<uint64_t, 64> epMask;
alignas(64) static array<array<int, 6>, 84> newDice;
alignas(64) static array<int, 216> Dice;
alignas(64) static array<array<int, 6>, 84> dicePiece;

static AI_FORCEINLINE char pieceChar(int pt) { return "pnbrqk"[pt]; }

static int diceCharOrder(char c) {
    switch (c) {
    case 'p': return 0;
    case 'n': return 1;
    case 'b': return 2;
    case 'r': return 3;
    case 'q': return 4;
    case 'k': return 5;
    default: return 99;
    }
}

static void initDice216AndDicePiece() {
    for (int d = 0; d < 84; ++d) {
        dicePiece[d].fill(0);
        const string& s = g_diceTable[d];
        for (char ch : s) {
            switch (ch) {
            case 'p': dicePiece[d][0]++; break;
            case 'n': dicePiece[d][1]++; break;
            case 'b': dicePiece[d][2]++; break;
            case 'r': dicePiece[d][3]++; break;
            case 'q': dicePiece[d][4]++; break;
            case 'k': dicePiece[d][5]++; break;
            default: break;
            }
        }
    }

    int out = 0;
    for (int i = 0; i < 6; ++i)
        for (int j = i; j < 6; ++j)
            for (int k = j; k < 6; ++k) {
                string s;
                s.push_back(pieceChar(i));
                s.push_back(pieceChar(j));
                s.push_back(pieceChar(k));

                auto it = g_diceIndex.find(s);
                int d = (it == g_diceIndex.end()) ? 0 : it->second;

                int mult;
                if (i == j && j == k) mult = 1;
                else if (i == j || j == k || i == k) mult = 3;
                else mult = 6;

                for (int t = 0; t < mult; ++t) Dice[out++] = d;
            }
    std::shuffle(Dice.begin(), Dice.end(), Random);
}

static void initDiceTable() {
    if (!g_diceTable.empty()) return;

    const string P = "pnbrqk";
    g_diceTable.reserve(84);
    g_diceTable.push_back("-"); // 0

    for (int i = 0; i < 6; ++i) { string s; s += P[i]; g_diceTable.push_back(s); }

    for (int i = 0; i < 6; ++i)
        for (int j = i; j < 6; ++j) {
            string s; s += P[i]; s += P[j];
            g_diceTable.push_back(s);
        }

    for (int i = 0; i < 6; ++i)
        for (int j = i; j < 6; ++j)
            for (int k = j; k < 6; ++k) {
                string s; s += P[i]; s += P[j]; s += P[k];
                g_diceTable.push_back(s);
            }

    for (int i = 0; i < (int)g_diceTable.size(); ++i) g_diceIndex[g_diceTable[i]] = i;

    g_diceMask.fill(0);
    for (int v = 0; v < 84; ++v) {
        uint8_t m = 0;
        for (char ch : g_diceTable[v]) {
            switch (ch) {
            case 'p': m |= (1u << 0); break;
            case 'n': m |= (1u << 1); break;
            case 'b': m |= (1u << 2); break;
            case 'r': m |= (1u << 3); break;
            case 'q': m |= (1u << 4); break;
            case 'k': m |= (1u << 5); break;
            default: break;
            }
        }
        g_diceMask[v] = m;
    }

    initDice216AndDicePiece();
}

static int diceFenToInt(string tok) {
    string s;
    for (char ch : tok) {
        ch = (char)tolower((unsigned char)ch);
        if (ch == '-' || ch == 'p' || ch == 'n' || ch == 'b' || ch == 'r' || ch == 'q' || ch == 'k') s.push_back(ch);
    }
    if (s.empty()) s = "-";
    if (s == "-") return 0;

    sort(s.begin(), s.end(), [](char a, char b) { return diceCharOrder(a) < diceCharOrder(b); });
    if ((int)s.size() > 3) s.resize(3);

    auto it = g_diceIndex.find(s);
    return (it == g_diceIndex.end()) ? 0 : it->second;
}

static string diceIntToFen(int v) {
    if (v < 0 || v >= (int)g_diceTable.size()) return "-";
    return g_diceTable[v];
}

static AI_FORCEINLINE uint8_t diceAllowedMaskFast(int diceVal) { return g_diceMask[diceVal]; }
static void initEpMaskAndNewDice() {
    for (int sq = 0; sq < 64; ++sq) {
        int r = sq >> 3;
        int f = sq & 7;

        uint64_t m = 0;

        if (r == 4) { // white rank5 -> rank6
            if (f > 0) m |= bit(sq + 7);
            if (f < 7) m |= bit(sq + 9);
        }
        if (r == 3) { // black rank4 -> rank3
            if (f > 0) m |= bit(sq - 9);
            if (f < 7) m |= bit(sq - 7);
        }

        epMask[sq] = m;
    }

    for (int d = 0; d < 84; ++d) {
        for (int pt = 0; pt < 6; ++pt) {
            string s = g_diceTable[d];
            if (s == "-") s.clear();

            char pc = pieceChar(pt);
            size_t pos = s.find(pc);
            if (pos != string::npos) s.erase(pos, 1);

            if (s.empty()) s = "-";

            auto it = g_diceIndex.find(s);
            newDice[d][pt] = (it == g_diceIndex.end()) ? 0 : it->second;
        }
    }
}



static uint64_t splitmix64(uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static uint64_t ZPiece[2][6][64];
static uint64_t ZSide;
static uint64_t ZCastle[16];
static uint64_t ZEp1[2][64];
static uint64_t ZEp2[64];
static uint64_t ZDice[84];

static void initZobrist() {
    uint64_t seed = 0xC0FFEE123456789ULL;

    for (int c = 0; c < 2; ++c)
        for (int p = 0; p < 6; ++p)
            for (int sq = 0; sq < 64; ++sq)
                ZPiece[c][p][sq] = splitmix64(seed);

    ZSide = splitmix64(seed);

    for (int i = 0; i < 16; ++i) ZCastle[i] = splitmix64(seed);

    for (int t = 0; t < 2; ++t)
        for (int sq = 0; sq < 64; ++sq)
            ZEp1[t][sq] = splitmix64(seed);

    for (int sq = 0; sq < 64; ++sq) ZEp2[sq] = splitmix64(seed);
    for (int i = 0; i < 84; ++i) ZDice[i] = splitmix64(seed);
}

static uint64_t computeKey(const Position& pos) {
    uint64_t k = 0;

    for (int c = 0; c < 2; ++c) {
        for (int p = 0; p < 6; ++p) {
            uint64_t bb = pos.piece[p] & pos.color[c];
            while (bb) {
                int sq = pop_lsb(bb);
                k ^= ZPiece[c][p][sq];
            }
        }
    }

    if (pos.side == 1) k ^= ZSide;
    k ^= ZCastle[pos.castle];

    for (int t = 0; t < 2; ++t) {
        uint64_t bb = pos.ep1[t];
        while (bb) {
            int sq = pop_lsb(bb);
            k ^= ZEp1[t][sq];
        }
    }

    uint64_t bb = pos.ep2;
    while (bb) {
        int sq = pop_lsb(bb);
        k ^= ZEp2[sq];
    }

    k ^= ZDice[pos.dice];
    return k;
}



static constexpr int NN_PIECE_PLANES = 12;
static constexpr int NN_EP1_PLANES = 2;
static constexpr int NN_EP2_PLANES = 1;
static constexpr int NN_CASTLE_PLANES = 4;
static constexpr int NN_DICE_PLANES = 18;
static constexpr int LEGACY_NN_DICE_PLANES = 6;

static constexpr int NN_SQ_PLANES = NN_PIECE_PLANES + NN_EP1_PLANES + NN_EP2_PLANES + NN_CASTLE_PLANES + NN_DICE_PLANES; // 37
static constexpr int LEGACY_NN_SQ_PLANES = NN_PIECE_PLANES + NN_EP1_PLANES + NN_EP2_PLANES + NN_CASTLE_PLANES + LEGACY_NN_DICE_PLANES; // 25
static constexpr int NN_INPUT_SIZE = NN_SQ_PLANES * 64; // 2368

using NNInput = array<float, NN_INPUT_SIZE>;

alignas(64) static array<float, 64> NN_PLANE0;
alignas(64) static array<float, 64> NN_PLANE1;

static void initNNConstPlanes() {
    NN_PLANE0.fill(0.0f);
    NN_PLANE1.fill(1.0f);
}

static AI_FORCEINLINE void copyPlane(NNInput& out, int plane, const float* src64) {
    memcpy(out.data() + plane * 64, src64, 64 * sizeof(float));
}
struct CanonGeom {
    int vflip = 0;          // 0 or 56
    bool hmirror = false;   // mirror files so king is always on the right

    AI_FORCEINLINE int sq(int s) const {
        s ^= vflip;         // side-to-move perspective
        if (hmirror) s ^= 7; // file mirror
        return s;
    }
};

static AI_FORCEINLINE CanonGeom canonicalGeom(const Position& pos) {
    CanonGeom g;
    g.vflip = pos.side ? 56 : 0;

    // own king in side-to-move perspective
    int ksq = ctz64(pos.piece[5] & pos.color[pos.side]);
    int kCanon = ksq ^ g.vflip;

    // if king is on files a..d, mirror so it ends up on e..h
    g.hmirror = ((kCanon & 7) < 4);
    return g;
}
static AI_FORCEINLINE void positionToNNInput(const Position& pos, NNInput& out) {
    memset(out.data(), 0, sizeof(out));

    const int usC = pos.side;
    const int themC = usC ^ 1;
    const CanonGeom cg = canonicalGeom(pos);

    float* outp = out.data();

    for (int pt = 0; pt < 6; ++pt) {
        {
            float* base = outp + (pt * 64);
            uint64_t bb = pos.piece[pt] & pos.color[usC];
            while (bb) {
                int sq = pop_lsb(bb);
                base[cg.sq(sq)] = 1.0f;
            }
        }
        {
            float* base = outp + ((6 + pt) * 64);
            uint64_t bb = pos.piece[pt] & pos.color[themC];
            while (bb) {
                int sq = pop_lsb(bb);
                base[cg.sq(sq)] = 1.0f;
            }
        }
    }

    {
        float* base = outp + (12 * 64);
        uint64_t bb = pos.ep1[usC];
        while (bb) {
            int sq = pop_lsb(bb);
            base[cg.sq(sq)] = 1.0f;
        }
    }
    {
        float* base = outp + (13 * 64);
        uint64_t bb = pos.ep1[themC];
        while (bb) {
            int sq = pop_lsb(bb);
            base[cg.sq(sq)] = 1.0f;
        }
    }

    {
        float* base = outp + (14 * 64);
        uint64_t bb = pos.ep2;
        while (bb) {
            int sq = pop_lsb(bb);
            base[cg.sq(sq)] = 1.0f;
        }
    }

    {
        // canonical castle planes:
        // 15 usQ, 16 usK, 17 themQ, 18 themK
        //
        // BUT when we mirror files to make king right,
        // Q/K semantics swap in canonical coordinates.
        int usQ = (usC == 0) ? 0 : 2;
        int usK = (usC == 0) ? 1 : 3;
        int themQ = (usC == 0) ? 2 : 0;
        int themK = (usC == 0) ? 3 : 1;

        auto putCastle = [&](int plane, int rookIdx) {
            if (((pos.castle >> rookIdx) & 1) && (unsigned)pos.rook[rookIdx] < 64u) {
                outp[plane * 64 + cg.sq(pos.rook[rookIdx])] = 1.0f;
            }
            };

        if (!cg.hmirror) {
            putCastle(15, usQ);
            putCastle(16, usK);
            putCastle(17, themQ);
            putCastle(18, themK);
        }
        else {
            // after file mirror, queenside/kingside swap in canonical view
            putCastle(15, usK);
            putCastle(16, usQ);
            putCastle(17, themK);
            putCastle(18, themQ);
        }
    }

    {
        const int d = pos.dice;
        for (int pt = 0; pt < 6; ++pt) {
            const int cnt = dicePiece[d][pt];
            for (int lvl = 0; lvl < 3; ++lvl) {
                copyPlane(out, 19 + pt * 3 + lvl, (cnt > lvl) ? NN_PLANE1.data() : NN_PLANE0.data());
            }
        }
    }
}



// Canonical policy index: CHW (plane-major), matches [B,73,8,8] NCHW flatten:
// k = plane*64 + sq (sq is "from-square after flip", 0..63)
static AI_FORCEINLINE int policyIndexCHWCanonical(int move, const Position& pos) {
    int from = move & 63;
    int to = (move >> 6) & 63;
    int promo = (move >> 12) & 7;

    const CanonGeom cg = canonicalGeom(pos);

    int rf = cg.sq(from);
    int rt = cg.sq(to);

    int fr = rf >> 3, ff = rf & 7;
    int tr = rt >> 3, tf = rt & 7;

    int dr = tr - fr;
    int df = tf - ff;

    int plane = 0;

    if (promo >= 1 && promo <= 3) {
        int dir = df + 1;        // -1->0, 0->1, +1->2
        int pGroup = 3 - promo;  // r(3)->0, b(2)->1, n(1)->2
        plane = 64 + pGroup * 3 + dir; // 64..72
    }
    else {
        if ((iabs(dr) == 2 && iabs(df) == 1) || (iabs(dr) == 1 && iabs(df) == 2)) {
            static constexpr int KNR[8] = { +2, +1, -1, -2, -2, -1, +1, +2 };
            static constexpr int KNF[8] = { +1, +2, +2, +1, -1, -2, -2, -1 };
            int kidx = 0;
            for (int i = 0; i < 8; ++i)
                if (dr == KNR[i] && df == KNF[i]) { kidx = i; break; }
            plane = 56 + kidx;
        }
        else {
            int dir = 0;
            int dist = 1;

            if (df == 0) { dist = iabs(dr); dir = (dr > 0) ? 0 : 4; }
            else if (dr == 0) { dist = iabs(df); dir = (df > 0) ? 2 : 6; }
            else {
                dist = iabs(dr);
                if (dr > 0 && df > 0) dir = 1;
                else if (dr < 0 && df > 0) dir = 3;
                else if (dr < 0 && df < 0) dir = 5;
                else dir = 7;
            }

            plane = dir * 7 + (dist - 1);
        }
    }

    return plane * 64 + rf;
}



alignas(64) static uint64_t KnightAtt[64];
alignas(64) static uint64_t KingAtt[64];

static void initLeaperAttacks() {
    for (int sq = 0; sq < 64; ++sq) {
        int r = sq >> 3, f = sq & 7;

        uint64_t n = 0;
        const int drN[8] = { +2,+2,+1,+1,-1,-1,-2,-2 };
        const int dfN[8] = { +1,-1,+2,-2,+2,-2,+1,-1 };
        for (int i = 0; i < 8; ++i) {
            int rr = r + drN[i], ff = f + dfN[i];
            if ((unsigned)rr < 8u && (unsigned)ff < 8u) n |= bit(rr * 8 + ff);
        }
        KnightAtt[sq] = n;

        uint64_t k = 0;
        for (int dr = -1; dr <= 1; ++dr)
            for (int df = -1; df <= 1; ++df) {
                if (dr == 0 && df == 0) continue;
                int rr = r + dr, ff = f + df;
                if ((unsigned)rr < 8u && (unsigned)ff < 8u) k |= bit(rr * 8 + ff);
            }
        KingAtt[sq] = k;
    }
}



static AI_FORCEINLINE uint64_t rookAttacksOTF(int sq, uint64_t occ) {
    uint64_t a = 0;
    int r = sq >> 3, f = sq & 7;

    for (int rr = r + 1; rr < 8; ++rr) { int s = rr * 8 + f; a |= bit(s); if (occ & bit(s)) break; }
    for (int rr = r - 1; rr >= 0; --rr) { int s = rr * 8 + f; a |= bit(s); if (occ & bit(s)) break; }
    for (int ff = f + 1; ff < 8; ++ff) { int s = r * 8 + ff; a |= bit(s); if (occ & bit(s)) break; }
    for (int ff = f - 1; ff >= 0; --ff) { int s = r * 8 + ff; a |= bit(s); if (occ & bit(s)) break; }

    return a;
}

static AI_FORCEINLINE uint64_t bishopAttacksOTF(int sq, uint64_t occ) {
    uint64_t a = 0;
    int r = sq >> 3, f = sq & 7;

    for (int rr = r + 1, ff = f + 1; rr < 8 && ff < 8; ++rr, ++ff) { int s = rr * 8 + ff; a |= bit(s); if (occ & bit(s)) break; }
    for (int rr = r + 1, ff = f - 1; rr < 8 && ff >= 0; ++rr, --ff) { int s = rr * 8 + ff; a |= bit(s); if (occ & bit(s)) break; }
    for (int rr = r - 1, ff = f + 1; rr >= 0 && ff < 8; --rr, ++ff) { int s = rr * 8 + ff; a |= bit(s); if (occ & bit(s)) break; }
    for (int rr = r - 1, ff = f - 1; rr >= 0 && ff >= 0; --rr, --ff) { int s = rr * 8 + ff; a |= bit(s); if (occ & bit(s)) break; }

    return a;
}

static AI_FORCEINLINE uint64_t rookMask(int sq) {
    uint64_t m = 0;
    int r = sq >> 3, f = sq & 7;
    for (int rr = r + 1; rr <= 6; ++rr) m |= bit(rr * 8 + f);
    for (int rr = r - 1; rr >= 1; --rr) m |= bit(rr * 8 + f);
    for (int ff = f + 1; ff <= 6; ++ff) m |= bit(r * 8 + ff);
    for (int ff = f - 1; ff >= 1; --ff) m |= bit(r * 8 + ff);
    return m;
}

static AI_FORCEINLINE uint64_t bishopMask(int sq) {
    uint64_t m = 0;
    int r = sq >> 3, f = sq & 7;
    for (int rr = r + 1, ff = f + 1; rr <= 6 && ff <= 6; ++rr, ++ff) m |= bit(rr * 8 + ff);
    for (int rr = r + 1, ff = f - 1; rr <= 6 && ff >= 1; ++rr, --ff) m |= bit(rr * 8 + ff);
    for (int rr = r - 1, ff = f + 1; rr >= 1 && ff <= 6; --rr, ++ff) m |= bit(rr * 8 + ff);
    for (int rr = r - 1, ff = f - 1; rr >= 1 && ff >= 1; --rr, --ff) m |= bit(rr * 8 + ff);
    return m;
}

static AI_FORCEINLINE uint64_t subsetFromIndex(uint32_t idx, uint64_t mask) {
    uint64_t occ = 0;
    while (mask) {
        int sq = pop_lsb(mask);
        if (idx & 1u) occ |= bit(sq);
        idx >>= 1;
    }
    return occ;
}

static constexpr uint64_t ROOK_MAGICS[64] = {
    0x8a80104000800020ULL, 0x140002000100040ULL, 0x2801880a0017001ULL, 0x100081001000420ULL,
    0x200020010080420ULL, 0x3001c0002010008ULL, 0x8480008002000100ULL, 0x2080088004402900ULL,
    0x800098204000ULL, 0x2024401000200040ULL, 0x100802000801000ULL, 0x120800800801000ULL,
    0x208808088000400ULL, 0x2802200800400ULL, 0x2200800100020080ULL, 0x801000060821100ULL,
    0x80044006422000ULL, 0x100808020004000ULL, 0x12108a0010204200ULL, 0x140848010000802ULL,
    0x481828014002800ULL, 0x8094004002004100ULL, 0x4010040010010802ULL, 0x20008806104ULL,
    0x100400080208000ULL, 0x2040002120081000ULL, 0x21200680100081ULL, 0x20100080080080ULL,
    0x2000a00200410ULL, 0x20080800400ULL, 0x80088400100102ULL, 0x80004600042881ULL,
    0x4040008040800020ULL, 0x440003000200801ULL, 0x4200011004500ULL, 0x188020010100100ULL,
    0x14800401802800ULL, 0x2080040080800200ULL, 0x124080204001001ULL, 0x200046502000484ULL,
    0x480400080088020ULL, 0x1000422010034000ULL, 0x30200100110040ULL, 0x100021010009ULL,
    0x2002080100110004ULL, 0x202008004008002ULL, 0x20020004010100ULL, 0x2048440040820001ULL,
    0x101002200408200ULL, 0x40802000401080ULL, 0x4008142004410100ULL, 0x2060820c0120200ULL,
    0x1001004080100ULL, 0x20c020080040080ULL, 0x2935610830022400ULL, 0x44440041009200ULL,
    0x280001040802101ULL, 0x2100190040002085ULL, 0x80c0084100102001ULL, 0x4024081001000421ULL,
    0x20030a0244872ULL, 0x12001008414402ULL, 0x2006104900a0804ULL, 0x1004081002402ULL
};

static constexpr uint64_t BISHOP_MAGICS[64] = {
    0x40040844404084ULL, 0x2004208a004208ULL, 0x10190041080202ULL, 0x108060845042010ULL,
    0x581104180800210ULL, 0x2112080446200010ULL, 0x1080820820060210ULL, 0x3c0808410220200ULL,
    0x4050404440404ULL, 0x21001420088ULL, 0x24d0080801082102ULL, 0x1020a0a020400ULL,
    0x40308200402ULL, 0x4011002100800ULL, 0x401484104104005ULL, 0x801010402020200ULL,
    0x400210c3880100ULL, 0x404022024108200ULL, 0x810018200204102ULL, 0x4002801a02003ULL,
    0x85040820080400ULL, 0x810102c808880400ULL, 0xe900410884800ULL, 0x8002020480840102ULL,
    0x220200865090201ULL, 0x2010100a02021202ULL, 0x152048408022401ULL, 0x20080002081110ULL,
    0x4001001021004000ULL, 0x800040400a011002ULL, 0xe4004081011002ULL, 0x1c004001012080ULL,
    0x8004200962a00220ULL, 0x8422100208500202ULL, 0x2000402200300c08ULL, 0x8646020080080080ULL,
    0x80020a0200100808ULL, 0x2010004880111000ULL, 0x623000a080011400ULL, 0x42008c0340209202ULL,
    0x209188240001000ULL, 0x400408a884001800ULL, 0x110400a6080400ULL, 0x1840060a44020800ULL,
    0x90080104000041ULL, 0x201011000808101ULL, 0x1a2208080504f080ULL, 0x8012020600211212ULL,
    0x500861011240000ULL, 0x180806108200800ULL, 0x4000020e01040044ULL, 0x300000261044000aULL,
    0x802241102020002ULL, 0x20906061210001ULL, 0x5a84841004010310ULL, 0x4010801011c04ULL,
    0xa010109502200ULL, 0x4a02012000ULL, 0x500201010098b028ULL, 0x8040002811040900ULL,
    0x28000010020204ULL, 0x6000020202d0240ULL, 0x8918844842082200ULL, 0x4010011029020020ULL
};

struct SliderPextTables {
    array<uint64_t, 64> rMask{}, bMask{};
    array<int, 64> rOff{}, bOff{};
    vector<uint64_t> rAtt;
    vector<uint64_t> bAtt;
};

struct SliderMagicTables {
    array<uint64_t, 64> rMask{}, bMask{};
    array<int, 64> rShift{}, bShift{};
    array<int, 64> rOff{}, bOff{};
    vector<uint64_t> rAtt;
    vector<uint64_t> bAtt;
};

static bool g_usePext = false;
static SliderPextTables g_pext;
static SliderMagicTables g_mag;

static void initSlidersPext() {
    g_pext.rAtt.clear();
    g_pext.bAtt.clear();

    size_t rTotal = 0, bTotal = 0;
    for (int sq = 0; sq < 64; ++sq) {
        rTotal += (size_t)1u << popcount64(rookMask(sq));
        bTotal += (size_t)1u << popcount64(bishopMask(sq));
    }
    g_pext.rAtt.reserve(rTotal);
    g_pext.bAtt.reserve(bTotal);

    int ro = 0, bo = 0;
    for (int sq = 0; sq < 64; ++sq) {
        uint64_t rm = rookMask(sq);
        uint64_t bm = bishopMask(sq);

        g_pext.rMask[sq] = rm;
        g_pext.bMask[sq] = bm;

        int rb = popcount64(rm);
        int bb = popcount64(bm);

        int rSize = 1 << rb;
        int bSize = 1 << bb;

        g_pext.rOff[sq] = ro;
        g_pext.bOff[sq] = bo;

        g_pext.rAtt.resize(ro + rSize);
        g_pext.bAtt.resize(bo + bSize);

        for (int i = 0; i < rSize; ++i) {
            uint64_t occ = subsetFromIndex((uint32_t)i, rm);
            g_pext.rAtt[ro + i] = rookAttacksOTF(sq, occ);
        }
        for (int i = 0; i < bSize; ++i) {
            uint64_t occ = subsetFromIndex((uint32_t)i, bm);
            g_pext.bAtt[bo + i] = bishopAttacksOTF(sq, occ);
        }

        ro += rSize;
        bo += bSize;
    }
}

static void initSlidersMagics() {
    g_mag.rAtt.clear();
    g_mag.bAtt.clear();

    size_t rTotal = 0, bTotal = 0;
    for (int sq = 0; sq < 64; ++sq) {
        rTotal += (size_t)1u << popcount64(rookMask(sq));
        bTotal += (size_t)1u << popcount64(bishopMask(sq));
    }
    g_mag.rAtt.reserve(rTotal);
    g_mag.bAtt.reserve(bTotal);

    int ro = 0, bo = 0;
    for (int sq = 0; sq < 64; ++sq) {
        uint64_t rm = rookMask(sq);
        uint64_t bm = bishopMask(sq);

        g_mag.rMask[sq] = rm;
        g_mag.bMask[sq] = bm;

        int rb = popcount64(rm);
        int bb = popcount64(bm);

        g_mag.rShift[sq] = 64 - rb;
        g_mag.bShift[sq] = 64 - bb;

        int rSize = 1 << rb;
        int bSize = 1 << bb;

        g_mag.rOff[sq] = ro;
        g_mag.bOff[sq] = bo;

        g_mag.rAtt.resize(ro + rSize);
        g_mag.bAtt.resize(bo + bSize);

        for (int i = 0; i < rSize; ++i) {
            uint64_t occ = subsetFromIndex((uint32_t)i, rm);
            uint64_t idx = (occ * ROOK_MAGICS[sq]) >> g_mag.rShift[sq];
            g_mag.rAtt[ro + (int)idx] = rookAttacksOTF(sq, occ);
        }
        for (int i = 0; i < bSize; ++i) {
            uint64_t occ = subsetFromIndex((uint32_t)i, bm);
            uint64_t idx = (occ * BISHOP_MAGICS[sq]) >> g_mag.bShift[sq];
            g_mag.bAtt[bo + (int)idx] = bishopAttacksOTF(sq, occ);
        }

        ro += rSize;
        bo += bSize;
    }
}

template<bool USE_PEXT>
static AI_FORCEINLINE uint64_t rookAttT(int sq, uint64_t occ) {
    if constexpr (USE_PEXT) {
#if HAVE_PEXT_INTRIN
        uint64_t idx = pext_u64_runtime(occ, g_pext.rMask[sq]);
        return g_pext.rAtt[g_pext.rOff[sq] + (int)idx];
#else
        return rookAttacksOTF(sq, occ);
#endif
    }
    else {
        uint64_t occ2 = occ & g_mag.rMask[sq];
        uint64_t idx = (occ2 * ROOK_MAGICS[sq]) >> g_mag.rShift[sq];
        return g_mag.rAtt[g_mag.rOff[sq] + (int)idx];
    }
}

template<bool USE_PEXT>
static AI_FORCEINLINE uint64_t bishopAttT(int sq, uint64_t occ) {
    if constexpr (USE_PEXT) {
#if HAVE_PEXT_INTRIN
        uint64_t idx = pext_u64_runtime(occ, g_pext.bMask[sq]);
        return g_pext.bAtt[g_pext.bOff[sq] + (int)idx];
#else
        return bishopAttacksOTF(sq, occ);
#endif
    }
    else {
        uint64_t occ2 = occ & g_mag.bMask[sq];
        uint64_t idx = (occ2 * BISHOP_MAGICS[sq]) >> g_mag.bShift[sq];
        return g_mag.bAtt[g_mag.bOff[sq] + (int)idx];
    }
}



static AI_FORCEINLINE int findKingSquare(const Position& pos, int colorIdx) {
    return ctz64(pos.piece[5] & pos.color[colorIdx]);
}

static void buildPathMask(const Position& pos, array<uint64_t, 4>& path, array<int, 64>& mask) {
    path = { 0ULL, 0ULL, 0ULL, 0ULL };
    mask.fill(0);

    int kingSq[2] = { findKingSquare(pos, 0), findKingSquare(pos, 1) };

    for (int i = 0; i < 4; ++i) {
        if (((pos.castle >> i) & 1) == 0) continue;

        const int rSq = pos.rook[i];
        const int kSq = kingSq[i / 2];

        int kDst = 2 + 4 * (i & 1) + 56 * (i >> 1);
        int rDst = 3 + 2 * (i & 1) + 56 * (i >> 1);

        int mn = rSq, mx = rSq;
        if (kSq < mn) mn = kSq;
        if (kDst < mn) mn = kDst;
        if (rDst < mn) mn = rDst;
        if (kSq > mx) mx = kSq;
        if (kDst > mx) mx = kDst;
        if (rDst > mx) mx = rDst;

        uint64_t bbpath = 0ULL;
        for (int sq = mn; sq <= mx; ++sq) bbpath |= bit(sq);

        bbpath &= ~bit(rSq);
        bbpath &= ~bit(kSq);
        path[i] = bbpath;

        mask[rSq] |= (1 << i);
        mask[kSq] |= (1 << i);
    }
}



static array<int, 8> genBackRank960() {
    array<int, 8> a;
    a.fill(-1);

    vector<int> freeSq;
    freeSq.reserve(8);
    for (int f = 0; f < 8; ++f) freeSq.push_back(f);

    auto eraseFree = [&](int f) {
        for (int i = 0; i < (int)freeSq.size(); ++i)
            if (freeSq[i] == f) { freeSq[i] = freeSq.back(); freeSq.pop_back(); return; }
        };

    vector<int> dark = { 0,2,4,6 };
    vector<int> light = { 1,3,5,7 };

    int b1 = dark[randInt(4)];
    int b2 = light[randInt(4)];
    a[b1] = 2; eraseFree(b1);
    a[b2] = 2; eraseFree(b2);

    {
        int qf = freeSq[randInt((int)freeSq.size())];
        a[qf] = 4;
        eraseFree(qf);
    }

    for (int t = 0; t < 2; ++t) {
        int nf = freeSq[randInt((int)freeSq.size())];
        a[nf] = 1;
        eraseFree(nf);
    }

    sort(freeSq.begin(), freeSq.end());
    a[freeSq[0]] = 3;
    a[freeSq[1]] = 5;
    a[freeSq[2]] = 3;

    return a;
}

static void placePiece(Position& pos, int colorIdx, int pt, int sq) {
    uint64_t b = bit(sq);
    pos.piece[pt] |= b;
    pos.color[colorIdx] |= b;
}

static void setRookCastlingSquaresFromBackRank(Position& pos, int colorIdx) {
    const uint64_t rankMask = (colorIdx == 0) ? RANK_1 : RANK_8;
    const int baseIndex = (colorIdx == 0) ? 0 : 2;

    uint64_t kingBB = pos.piece[5] & pos.color[colorIdx] & rankMask;
    uint64_t rooksBB = pos.piece[3] & pos.color[colorIdx] & rankMask;

    int ksq = ctz64(kingBB);
    int kf = ksq & 7;

    int qR = -1, kR = -1;
    uint64_t tmp = rooksBB;
    while (tmp) {
        int rsq = pop_lsb(tmp);
        int rf = rsq & 7;
        if (rf < kf) qR = rsq;
        else         kR = rsq;
    }

    pos.rook[baseIndex + 0] = qR;
    pos.rook[baseIndex + 1] = kR;
}

static void chess960(Position& pos, array<uint64_t, 4>& path, array<int, 64>& mask) {
    pos.color = { 0ULL, 0ULL };
    pos.piece = { 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL };

    pos.side = 1;
    pos.ep1 = { 0ULL, 0ULL };
    pos.ep2 = 0ULL;
    pos.castle = 15;
    pos.dice = 0;
    pos.rook = { -1, -1, -1, -1 };
    pos.key = 0;

    for (int f = 0; f < 8; ++f) {
        placePiece(pos, 0, 0, 8 + f);
        placePiece(pos, 1, 0, 48 + f);
    }

    array<int, 8> w = genBackRank960();
    array<int, 8> b = genBackRank960();

    for (int f = 0; f < 8; ++f) {
        placePiece(pos, 0, w[f], 0 + f);
        placePiece(pos, 1, b[f], 56 + f);
    }

    setRookCastlingSquaresFromBackRank(pos, 0);
    setRookCastlingSquaresFromBackRank(pos, 1);

    buildPathMask(pos, path, mask);
    pos.key = computeKey(pos);
}



static void parseBoard(const string& board, Position& pos) {
    pos.color = { 0ULL, 0ULL };
    pos.piece = { 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL };

    int rank = 7;
    int file = 0;

    for (char ch : board) {
        if (ch == '/') { --rank; file = 0; continue; }
        if (isdigit((unsigned char)ch)) { file += (ch - '0'); continue; }

        int sq = rank * 8 + file;
        ++file;

        bool isWhite = isupper((unsigned char)ch);
        char pc = (char)tolower((unsigned char)ch);

        int p = -1;
        if (pc == 'p') p = 0;
        else if (pc == 'n') p = 1;
        else if (pc == 'b') p = 2;
        else if (pc == 'r') p = 3;
        else if (pc == 'q') p = 4;
        else if (pc == 'k') p = 5;

        if (p >= 0) {
            pos.piece[p] |= bit(sq);
            pos.color[isWhite ? 0 : 1] |= bit(sq);
        }
    }
}

static uint64_t parseSquaresTokenToBB(const string& tok) {
    if (tok == "-" || tok.empty()) return 0ULL;
    uint64_t bb = 0ULL;
    for (size_t i = 0; i + 1 < tok.size(); i += 2) {
        int sq = sqFromName2(tok[i], tok[i + 1]);
        if (sq >= 0) bb |= bit(sq);
    }
    return bb;
}

static void fenToPositionPathMask(const string& fen, Position& pos,
    array<uint64_t, 4>& path, array<int, 64>& mask) {
    pos.side = 0;
    pos.ep1 = { 0ULL, 0ULL };
    pos.ep2 = 0ULL;
    pos.rook = { -1, -1, -1, -1 };
    pos.castle = 0;
    pos.dice = 0;
    pos.key = 0;

    vector<string> t;
    {
        istringstream iss(fen);
        string s;
        while (iss >> s) t.push_back(s);
    }

    if (t.size() < 6) {
        if (!t.empty()) parseBoard(t[0], pos);
        buildPathMask(pos, path, mask);
        pos.key = computeKey(pos);
        return;
    }

    parseBoard(t[0], pos);
    pos.side = (t[1].size() && (t[1][0] == 'b' || t[1][0] == 'B')) ? 1 : 0;

    pos.ep1 = { 0ULL, 0ULL };
    if (t[2] != "-" && !t[2].empty()) {
        for (size_t i = 0; i + 1 < t[2].size(); i += 2) {
            char f = t[2][i];
            char r = t[2][i + 1];
            int sq = sqFromName2(f, r);
            if (sq < 0) continue;
            if (r == '6') pos.ep1[0] |= bit(sq);
            else if (r == '3') pos.ep1[1] |= bit(sq);
        }
    }

    pos.ep2 = parseSquaresTokenToBB(t[3]);

    string cTok = t[4];
    if (cTok == "-") cTok = "----";
    string cc;
    for (char ch : cTok) {
        ch = (char)tolower((unsigned char)ch);
        if (ch == '-' || (ch >= 'a' && ch <= 'h')) cc.push_back(ch);
    }
    if ((int)cc.size() >= 4) cc = cc.substr(0, 4);
    if ((int)cc.size() < 4) cc += string(4 - cc.size(), '-');

    pos.rook = { -1, -1, -1, -1 };
    pos.castle = 0;
    for (int i = 0; i < 4; ++i) {
        char f = cc[i];
        if (f == '-') continue;
        int fi = fileIndex(f);
        int sq = fi + 56 * (i >> 1);
        pos.rook[i] = sq;
        pos.castle |= (1 << i);
    }

    pos.dice = diceFenToInt(t[5]);

    buildPathMask(pos, path, mask);
    pos.key = computeKey(pos);
}

static AI_FORCEINLINE char pieceAtChar(const Position& pos, int sq) {
    uint64_t b = bit(sq);

    int pt = -1;
    for (int p = 0; p < 6; ++p) {
        if (pos.piece[p] & b) { pt = p; break; }
    }
    if (pt < 0) return '.';

    char c = "pnbrqk"[pt];
    bool isWhite = (pos.color[0] & b) != 0;
    return isWhite ? (char)toupper((unsigned char)c) : c;
}
static void printBoardViz(const Position& pos) {
    for (int r = 7; r >= 0; --r) {
        cout << "| ";
        for (int f = 0; f < 8; ++f) {
            int sq = r * 8 + f;
            uint64_t b = bit(sq);

            char ch = pieceAtChar(pos, sq);



            cout << ch << (f == 7 ? "" : " ");
        }
        cout << " |\n";
    }


}

static void printPositionPathMask(const Position& pos,
    const array<uint64_t, 4>& path,
    const array<int, 64>& mask) {
    cout << "color[0] " << bbToSquares(pos.color[0]) << "\n";
    cout << "color[1] " << bbToSquares(pos.color[1]) << "\n";
    for (int pt = 0; pt < 6; ++pt) cout << "piece[" << pt << "] " << bbToSquares(pos.piece[pt]) << "\n";
    cout << "side " << pos.side << "\n";
    cout << "ep1[0] " << bbToSquares(pos.ep1[0]) << "\n";
    cout << "ep1[1] " << bbToSquares(pos.ep1[1]) << "\n";
    cout << "ep2 " << bbToSquares(pos.ep2) << "\n";
    cout << "rook[0] " << sqName(pos.rook[0]) << "\n";
    cout << "rook[1] " << sqName(pos.rook[1]) << "\n";
    cout << "rook[2] " << sqName(pos.rook[2]) << "\n";
    cout << "rook[3] " << sqName(pos.rook[3]) << "\n";
    cout << "castle " << pos.castle << "\n";
    cout << "dice " << diceIntToFen(pos.dice) << "\n";
    cout << "key " << pos.key << "\n";
    for (int i = 0; i < 4; ++i) cout << "path[" << i << "] " << bbToSquares(path[i]) << "\n";
    cout << "mask\n";
    for (int r = 7; r >= 0; --r) {
        for (int f = 0; f < 8; ++f) {
            int sq = r * 8 + f;
            if (f == 0) cout << mask[sq];
            else        cout << ' ' << mask[sq];
        }
        cout << "\n";
    }
}



#define ADD_MOVE_FAST(FROM,TO,PROMO) (*outp++ = ((FROM) | ((TO) << 6) | ((PROMO) << 12)))

template<bool USE_PEXT, int SIDE>
static AI_FORCEINLINE AI_HOT void genMovesSideT(const Position& pos, const array<uint64_t, 4>& path, MoveList& ml) {
    int* __restrict outp = ml.m;

    constexpr int THEM = SIDE ^ 1;

    const uint64_t us = pos.color[SIDE];
    const uint64_t them = pos.color[THEM];
    const uint64_t occ = us | them;

    const uint64_t notUs = ~us;
    const uint64_t empty = ~occ;

    const uint8_t allow = diceAllowedMaskFast(pos.dice);
    if (AI_UNLIKELY(!allow)) { ml.n = 0; return; }

    const int ksq = ctz64(pos.piece[5] & us);

    // pawns
    if (allow & (1u << 0)) {
        const uint64_t pawns = pos.piece[0] & us;
        if (pawns) {
            if constexpr (SIDE == 0) {
                uint64_t toBB = (pawns << 8) & empty;
                uint64_t prom = toBB & RANK_8;
                uint64_t nonp = toBB & ~RANK_8;

                while (nonp) { int to = pop_lsb(nonp); ADD_MOVE_FAST(to - 8, to, 0); }
                while (prom) {
                    int to = pop_lsb(prom);
                    int from = to - 8;
                    ADD_MOVE_FAST(from, to, 1);
                    ADD_MOVE_FAST(from, to, 2);
                    ADD_MOVE_FAST(from, to, 3);
                    ADD_MOVE_FAST(from, to, 4);
                }

                uint64_t one = ((pawns & RANK_2) << 8) & empty;
                uint64_t two = (one << 8) & empty;
                while (two) { int to = pop_lsb(two); ADD_MOVE_FAST(to - 16, to, 0); }

                uint64_t capL = ((pawns << 7) & ~FILE_H) & them;
                uint64_t capR = ((pawns << 9) & ~FILE_A) & them;

                const uint64_t enemyKing = pos.piece[5] & them;
                uint64_t capLp = capL & RANK_8 & ~enemyKing;
                uint64_t capLn = (capL & ~RANK_8) | (capL & RANK_8 & enemyKing);
                uint64_t capRp = capR & RANK_8 & ~enemyKing;
                uint64_t capRn = (capR & ~RANK_8) | (capR & RANK_8 & enemyKing);

                while (capLn) { int to = pop_lsb(capLn); ADD_MOVE_FAST(to - 7, to, 0); }
                while (capRn) { int to = pop_lsb(capRn); ADD_MOVE_FAST(to - 9, to, 0); }

                while (capLp) {
                    int to = pop_lsb(capLp);
                    int from = to - 7;
                    ADD_MOVE_FAST(from, to, 1);
                    ADD_MOVE_FAST(from, to, 2);
                    ADD_MOVE_FAST(from, to, 3);
                    ADD_MOVE_FAST(from, to, 4);
                }
                while (capRp) {
                    int to = pop_lsb(capRp);
                    int from = to - 9;
                    ADD_MOVE_FAST(from, to, 1);
                    ADD_MOVE_FAST(from, to, 2);
                    ADD_MOVE_FAST(from, to, 3);
                    ADD_MOVE_FAST(from, to, 4);
                }

                uint64_t epTo = pos.ep1[0] & empty;
                if (epTo) {
                    const uint64_t enemyPawns = pos.piece[0] & them;
                    const uint64_t capMaskTo = (enemyPawns << 8);

                    uint64_t toL = epTo & capMaskTo & ~FILE_H;
                    uint64_t fromL = (toL >> 7) & pawns & ~pos.ep2;
                    while (fromL) { int from = pop_lsb(fromL); ADD_MOVE_FAST(from, from + 7, 0); }

                    uint64_t toR = epTo & capMaskTo & ~FILE_A;
                    uint64_t fromR = (toR >> 9) & pawns & ~pos.ep2;
                    while (fromR) { int from = pop_lsb(fromR); ADD_MOVE_FAST(from, from + 9, 0); }
                }
            }
            else {
                uint64_t toBB = (pawns >> 8) & empty;
                uint64_t prom = toBB & RANK_1;
                uint64_t nonp = toBB & ~RANK_1;

                while (nonp) { int to = pop_lsb(nonp); ADD_MOVE_FAST(to + 8, to, 0); }
                while (prom) {
                    int to = pop_lsb(prom);
                    int from = to + 8;
                    ADD_MOVE_FAST(from, to, 1);
                    ADD_MOVE_FAST(from, to, 2);
                    ADD_MOVE_FAST(from, to, 3);
                    ADD_MOVE_FAST(from, to, 4);
                }

                uint64_t one = ((pawns & RANK_7) >> 8) & empty;
                uint64_t two = (one >> 8) & empty;
                while (two) { int to = pop_lsb(two); ADD_MOVE_FAST(to + 16, to, 0); }

                uint64_t capL = ((pawns >> 9) & ~FILE_H) & them;
                uint64_t capR = ((pawns >> 7) & ~FILE_A) & them;

                const uint64_t enemyKing = pos.piece[5] & them;
                uint64_t capLp = capL & RANK_1 & ~enemyKing;
                uint64_t capLn = (capL & ~RANK_1) | (capL & RANK_1 & enemyKing);
                uint64_t capRp = capR & RANK_1 & ~enemyKing;
                uint64_t capRn = (capR & ~RANK_1) | (capR & RANK_1 & enemyKing);

                while (capLn) { int to = pop_lsb(capLn); ADD_MOVE_FAST(to + 9, to, 0); }
                while (capRn) { int to = pop_lsb(capRn); ADD_MOVE_FAST(to + 7, to, 0); }

                while (capLp) {
                    int to = pop_lsb(capLp);
                    int from = to + 9;
                    ADD_MOVE_FAST(from, to, 1);
                    ADD_MOVE_FAST(from, to, 2);
                    ADD_MOVE_FAST(from, to, 3);
                    ADD_MOVE_FAST(from, to, 4);
                }
                while (capRp) {
                    int to = pop_lsb(capRp);
                    int from = to + 7;
                    ADD_MOVE_FAST(from, to, 1);
                    ADD_MOVE_FAST(from, to, 2);
                    ADD_MOVE_FAST(from, to, 3);
                    ADD_MOVE_FAST(from, to, 4);
                }

                uint64_t epTo = pos.ep1[1] & empty;
                if (epTo) {
                    const uint64_t enemyPawns = pos.piece[0] & them;
                    const uint64_t capMaskTo = (enemyPawns >> 8);

                    uint64_t toL = epTo & capMaskTo & ~FILE_H;
                    uint64_t fromL = (toL << 9) & pawns & ~pos.ep2;
                    while (fromL) { int from = pop_lsb(fromL); ADD_MOVE_FAST(from, from - 9, 0); }

                    uint64_t toR = epTo & capMaskTo & ~FILE_A;
                    uint64_t fromR = (toR << 7) & pawns & ~pos.ep2;
                    while (fromR) { int from = pop_lsb(fromR); ADD_MOVE_FAST(from, from - 7, 0); }
                }
            }
        }
    }

    // knights
    if (allow & (1u << 1)) {
        uint64_t bb = pos.piece[1] & us;
        while (bb) {
            int from = pop_lsb(bb);
            uint64_t targets = KnightAtt[from] & notUs;
            while (targets) { int to = pop_lsb(targets); ADD_MOVE_FAST(from, to, 0); }
        }
    }

    // bishops
    if (allow & (1u << 2)) {
        uint64_t bb = pos.piece[2] & us;
        while (bb) {
            int from = pop_lsb(bb);
            uint64_t targets = bishopAttT<USE_PEXT>(from, occ) & notUs;
            while (targets) { int to = pop_lsb(targets); ADD_MOVE_FAST(from, to, 0); }
        }
    }

    // rooks
    if (allow & (1u << 3)) {
        uint64_t bb = pos.piece[3] & us;
        while (bb) {
            int from = pop_lsb(bb);
            uint64_t targets = rookAttT<USE_PEXT>(from, occ) & notUs;
            while (targets) { int to = pop_lsb(targets); ADD_MOVE_FAST(from, to, 0); }
        }
    }

    // queens
    if (allow & (1u << 4)) {
        uint64_t bb = pos.piece[4] & us;
        while (bb) {
            int from = pop_lsb(bb);
            uint64_t targets = (rookAttT<USE_PEXT>(from, occ) | bishopAttT<USE_PEXT>(from, occ)) & notUs;
            while (targets) { int to = pop_lsb(targets); ADD_MOVE_FAST(from, to, 0); }
        }
    }

    // king
    if (allow & (1u << 5)) {
        uint64_t targets = KingAtt[ksq] & notUs;
        while (targets) { int to = pop_lsb(targets); ADD_MOVE_FAST(ksq, to, 0); }
    }

    // castling last
    if ((allow & ((1u << 3) | (1u << 5))) == ((1u << 3) | (1u << 5))) {
        const int base = SIDE * 2;
        for (int i = base; i < base + 2; ++i) {
            if (((pos.castle >> i) & 1) == 0) continue;
            const int rsq = pos.rook[i];
            if ((path[i] & occ) != 0) continue;
            ADD_MOVE_FAST(ksq, rsq, 0);
        }
    }

    ml.n = (int)(outp - ml.m);
}

#undef ADD_MOVE_FAST

static AI_FORCEINLINE AI_HOT void genMoves(const Position& pos, const array<uint64_t, 4>& path, MoveList& ml) {
    if (g_usePext) {
        if (pos.side == 0) genMovesSideT<true, 0>(pos, path, ml);
        else               genMovesSideT<true, 1>(pos, path, ml);
    }
    else {
        if (pos.side == 0) genMovesSideT<false, 0>(pos, path, ml);
        else               genMovesSideT<false, 1>(pos, path, ml);
    }
}

#define RETURN_MOVE_FAST(FROM,TO,PROMO) return ((FROM) | ((TO) << 6) | ((PROMO) << 12))

template<bool USE_PEXT, int SIDE>
static AI_FORCEINLINE AI_HOT int genFirstSideT(const Position& pos, const array<uint64_t, 4>& path) {
    constexpr int THEM = SIDE ^ 1;

    const uint64_t us = pos.color[SIDE];
    const uint64_t them = pos.color[THEM];
    const uint64_t occ = us | them;

    const uint64_t notUs = ~us;
    const uint64_t empty = ~occ;

    const uint8_t allow = diceAllowedMaskFast(pos.dice);
    if (AI_UNLIKELY(!allow)) return 0;

    const int ksq = ctz64(pos.piece[5] & us);

    if (allow & (1u << 0)) {
        const uint64_t pawns = pos.piece[0] & us;
        if (pawns) {
            if constexpr (SIDE == 0) {
                uint64_t toBB = (pawns << 8) & empty;
                uint64_t prom = toBB & RANK_8;
                uint64_t nonp = toBB & ~RANK_8;
                if (nonp) { int to = ctz64(nonp); RETURN_MOVE_FAST(to - 8, to, 0); }
                if (prom) { int to = ctz64(prom); int from = to - 8; RETURN_MOVE_FAST(from, to, 1); }
                uint64_t one = ((pawns & RANK_2) << 8) & empty;
                uint64_t two = (one << 8) & empty;
                if (two) { int to = ctz64(two); RETURN_MOVE_FAST(to - 16, to, 0); }

                uint64_t capL = ((pawns << 7) & ~FILE_H) & them;
                uint64_t capR = ((pawns << 9) & ~FILE_A) & them;

                const uint64_t enemyKing = pos.piece[5] & them;
                uint64_t capLp = capL & RANK_8 & ~enemyKing;
                uint64_t capLn = (capL & ~RANK_8) | (capL & RANK_8 & enemyKing);
                uint64_t capRp = capR & RANK_8 & ~enemyKing;
                uint64_t capRn = (capR & ~RANK_8) | (capR & RANK_8 & enemyKing);

                if (capLn) { int to = ctz64(capLn); RETURN_MOVE_FAST(to - 7, to, 0); }
                if (capRn) { int to = ctz64(capRn); RETURN_MOVE_FAST(to - 9, to, 0); }
                if (capLp) { int to = ctz64(capLp); int from = to - 7; RETURN_MOVE_FAST(from, to, 1); }
                if (capRp) { int to = ctz64(capRp); int from = to - 9; RETURN_MOVE_FAST(from, to, 1); }

                uint64_t epTo = pos.ep1[0] & empty;
                if (epTo) {
                    const uint64_t enemyPawns = pos.piece[0] & them;
                    const uint64_t capMaskTo = (enemyPawns << 8);

                    uint64_t toL = epTo & capMaskTo & ~FILE_H;
                    uint64_t fromL = (toL >> 7) & pawns & ~pos.ep2;
                    if (fromL) { int from = ctz64(fromL); RETURN_MOVE_FAST(from, from + 7, 0); }

                    uint64_t toR = epTo & capMaskTo & ~FILE_A;
                    uint64_t fromR = (toR >> 9) & pawns & ~pos.ep2;
                    if (fromR) { int from = ctz64(fromR); RETURN_MOVE_FAST(from, from + 9, 0); }
                }
            }
            else {
                uint64_t toBB = (pawns >> 8) & empty;
                uint64_t prom = toBB & RANK_1;
                uint64_t nonp = toBB & ~RANK_1;
                if (nonp) { int to = ctz64(nonp); RETURN_MOVE_FAST(to + 8, to, 0); }
                if (prom) { int to = ctz64(prom); int from = to + 8; RETURN_MOVE_FAST(from, to, 1); }
                uint64_t one = ((pawns & RANK_7) >> 8) & empty;
                uint64_t two = (one >> 8) & empty;
                if (two) { int to = ctz64(two); RETURN_MOVE_FAST(to + 16, to, 0); }

                uint64_t capL = ((pawns >> 9) & ~FILE_H) & them;
                uint64_t capR = ((pawns >> 7) & ~FILE_A) & them;

                const uint64_t enemyKing = pos.piece[5] & them;
                uint64_t capLp = capL & RANK_1 & ~enemyKing;
                uint64_t capLn = (capL & ~RANK_1) | (capL & RANK_1 & enemyKing);
                uint64_t capRp = capR & RANK_1 & ~enemyKing;
                uint64_t capRn = (capR & ~RANK_1) | (capR & RANK_1 & enemyKing);

                if (capLn) { int to = ctz64(capLn); RETURN_MOVE_FAST(to + 9, to, 0); }
                if (capRn) { int to = ctz64(capRn); RETURN_MOVE_FAST(to + 7, to, 0); }
                if (capLp) { int to = ctz64(capLp); int from = to + 9; RETURN_MOVE_FAST(from, to, 1); }
                if (capRp) { int to = ctz64(capRp); int from = to + 7; RETURN_MOVE_FAST(from, to, 1); }

                uint64_t epTo = pos.ep1[1] & empty;
                if (epTo) {
                    const uint64_t enemyPawns = pos.piece[0] & them;
                    const uint64_t capMaskTo = (enemyPawns >> 8);

                    uint64_t toL = epTo & capMaskTo & ~FILE_H;
                    uint64_t fromL = (toL << 9) & pawns & ~pos.ep2;
                    if (fromL) { int from = ctz64(fromL); RETURN_MOVE_FAST(from, from - 9, 0); }

                    uint64_t toR = epTo & capMaskTo & ~FILE_A;
                    uint64_t fromR = (toR << 7) & pawns & ~pos.ep2;
                    if (fromR) { int from = ctz64(fromR); RETURN_MOVE_FAST(from, from - 7, 0); }
                }
            }
        }
    }

    if (allow & (1u << 1)) {
        uint64_t bb = pos.piece[1] & us;
        while (bb) {
            int from = pop_lsb(bb);
            uint64_t targets = KnightAtt[from] & notUs;
            if (targets) { int to = ctz64(targets); RETURN_MOVE_FAST(from, to, 0); }
        }
    }

    if (allow & (1u << 2)) {
        uint64_t bb = pos.piece[2] & us;
        while (bb) {
            int from = pop_lsb(bb);
            uint64_t targets = bishopAttT<USE_PEXT>(from, occ) & notUs;
            if (targets) { int to = ctz64(targets); RETURN_MOVE_FAST(from, to, 0); }
        }
    }

    if (allow & (1u << 3)) {
        uint64_t bb = pos.piece[3] & us;
        while (bb) {
            int from = pop_lsb(bb);
            uint64_t targets = rookAttT<USE_PEXT>(from, occ) & notUs;
            if (targets) { int to = ctz64(targets); RETURN_MOVE_FAST(from, to, 0); }
        }
    }

    if (allow & (1u << 4)) {
        uint64_t bb = pos.piece[4] & us;
        while (bb) {
            int from = pop_lsb(bb);
            uint64_t targets = (rookAttT<USE_PEXT>(from, occ) | bishopAttT<USE_PEXT>(from, occ)) & notUs;
            if (targets) { int to = ctz64(targets); RETURN_MOVE_FAST(from, to, 0); }
        }
    }

    if (allow & (1u << 5)) {
        uint64_t targets = KingAtt[ksq] & notUs;
        if (targets) { int to = ctz64(targets); RETURN_MOVE_FAST(ksq, to, 0); }
    }

    if ((allow & ((1u << 3) | (1u << 5))) == ((1u << 3) | (1u << 5))) {
        const int base = SIDE * 2;
        for (int i = base; i < base + 2; ++i) {
            if (((pos.castle >> i) & 1) == 0) continue;
            const int rsq = pos.rook[i];
            if ((path[i] & occ) != 0) continue;
            RETURN_MOVE_FAST(ksq, rsq, 0);
        }
    }

    return 0;
}

#undef RETURN_MOVE_FAST

static AI_FORCEINLINE AI_HOT int genFirst(const Position& pos, const array<uint64_t, 4>& path) {
    if (g_usePext) {
        if (pos.side == 0) return genFirstSideT<true, 0>(pos, path);
        else               return genFirstSideT<true, 1>(pos, path);
    }
    else {
        if (pos.side == 0) return genFirstSideT<false, 0>(pos, path);
        else               return genFirstSideT<false, 1>(pos, path);
    }
}



void makeMove(Position& pos, const array<int, 64>& mask, int move) {
    int from = move & 63;
    int to = move >> 6 & 63;
    uint64_t f = bit(from);
    uint64_t t = bit(to);
    uint64_t ft = f | t;
    int side = pos.side;
    int opp = !side;
    uint64_t& us = pos.color[side];
    uint64_t& them = pos.color[opp];
    uint64_t us2 = us & t;
    uint64_t them2 = them & t;
    array<uint64_t, 6>& piece = pos.piece;
    int moving = Piece(pos, from);
    if (them2) {
        int cap = Piece(pos, to);
        them ^= t;
        piece[cap] ^= t;
        pos.key ^= ZPiece[opp][cap][to];
    }
    us ^= ft;
    piece[moving] ^= ft;
    pos.key ^= ZPiece[side][moving][from] ^ ZPiece[side][moving][to];
    pos.key ^= ZCastle[pos.castle];
    pos.castle &= ~(mask[from] | mask[to]);
    pos.key ^= ZCastle[pos.castle];
    pos.key ^= ZDice[pos.dice];
    pos.dice = newDice[pos.dice][moving];
    pos.key ^= ZDice[pos.dice];
    if (moving == 0) {
        if (pos.ep1[side] & epMask[to] && (pos.ep2 & t) == 0) {
            pos.ep2 |= t;
            pos.key ^= ZEp2[to];
            return;
        }
        if ((from ^ to) == 16) {
            int ep = (from + to) / 2;
            pos.ep1[opp] |= bit(ep);
            pos.key ^= ZEp1[opp][ep];
            return;
        }
        if (to <= 7 || to >= 56) {
            int promo = move >> 12;
            piece[0] ^= t;
            piece[promo] ^= t;
            pos.key ^= ZPiece[side][0][to] ^ ZPiece[side][promo][to];
            return;
        }
        if (((to - from) & 7) && them2 == 0) {
            int cap = to - 8 + (side << 4);
            uint64_t c = bit(cap);
            them ^= c;
            piece[0] ^= c;
            pos.key ^= ZPiece[opp][0][cap];
            return;
        }
    }
    if (us2 == 0)return;
    int dir = to > from;
    int y = 56 * side;
    int rook = 3 + (dir << 1) + y;
    int king = 2 + (dir << 2) + y;
    uint64_t r = bit(rook);
    uint64_t k = bit(king);
    us ^= r ^ k;
    piece[3] ^= t ^ r;
    piece[5] ^= t ^ k;
    pos.key ^= ZPiece[side][3][to] ^ ZPiece[side][3][rook] ^ ZPiece[side][5][to] ^ ZPiece[side][5][king];
    pos.key ^= ZDice[pos.dice];
    pos.dice = newDice[pos.dice][3];
    pos.key ^= ZDice[pos.dice];
}

static void makeRandomWithRolledDice(Position& pos, TTNode* node, int rolledDice) {
    while (pos.ep1[pos.side]) {
        int sq = pop_lsb(pos.ep1[pos.side]);
        pos.key ^= ZEp1[pos.side][sq];
    }
    while (pos.ep2) {
        int sq = pop_lsb(pos.ep2);
        pos.key ^= ZEp2[sq];
    }
    pos.side = !pos.side;
    pos.key ^= ZSide;
    uint32_t visits = node ? node->visits.load(std::memory_order_relaxed) : 0;
    uint64_t pawns = pos.color[pos.side] & pos.piece[0];
    int dist = 6;
    pos.key ^= ZDice[pos.dice];
    pos.dice = rolledDice;
    if (pawns) {
        if (pos.side == 0)dist = clz64(pawns) >> 3;   // MSVC-safe
        else dist = ctz64(pawns) >> 3;            // MSVC-safe
    }
    for (int i = 0; i < 5; i++)
        while (dicePiece[pos.dice][i] && (pos.color[pos.side] & pos.piece[i]) == 0 && dist > dicePiece[pos.dice][0])
            pos.dice = newDice[pos.dice][i];
    pos.key ^= ZDice[pos.dice];
}

void makeRandom(Position& pos, TTNode* node) {
    makeRandomWithRolledDice(pos, node, Dice[Range(Random)]);
}


static AI_FORCEINLINE uint32_t mix32From64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return (uint32_t)x ^ (uint32_t)(x >> 32);
}

// 216 = 2^3 * 3^3
// So that (base + step * k) iterates through all residue classes, step must be coprime with 216,
// i.e., it must NOT be divisible by 2 or 3.
static AI_FORCEINLINE uint32_t normalizeStepMod216(uint32_t s) {
    s %= 216u;
    if (s == 0u) s = 1u;

    while ((s & 1u) == 0u || (s % 3u) == 0u) {
        ++s;
        if (s >= 216u) s -= 216u;
        if (s == 0u) s = 1u;
    }
    return s;
}

static AI_FORCEINLINE uint32_t deterministicDiceBase216(uint64_t key) {
    return mix32From64(key ^ 0x9E3779B97F4A7C15ULL) % 216u;
}

static AI_FORCEINLINE uint32_t deterministicDiceStep216(uint64_t key) {
    uint32_t s = mix32From64(key ^ 0xD1B54A32D192ED03ULL);
    return normalizeStepMod216(s);
}

void makeRandomDeterministic(Position& pos, TTNode* node) {
    // fallback: if there is no node, use the old random behavior
    if (!node) {
        makeRandom(pos, node);
        return;
    }

    while (pos.ep1[pos.side]) {
        int sq = pop_lsb(pos.ep1[pos.side]);
        pos.key ^= ZEp1[pos.side][sq];
    }
    while (pos.ep2) {
        int sq = pop_lsb(pos.ep2);
        pos.key ^= ZEp2[sq];
    }

    pos.side = !pos.side;
    pos.key ^= ZSide;

    const uint32_t cursor = node->chanceCursor.fetch_add(1, std::memory_order_relaxed);
    const uint32_t base = deterministicDiceBase216(node->key);
    const uint32_t step = deterministicDiceStep216(node->key);

    const uint32_t idx = (uint32_t)((base + (uint64_t)step * (uint64_t)cursor) % 216u);

    uint64_t pawns = pos.color[pos.side] & pos.piece[0];
    int dist = 6;

    pos.key ^= ZDice[pos.dice];
    pos.dice = Dice[(size_t)idx];

    if (pawns) {
        if (pos.side == 0) dist = clz64(pawns) >> 3;
        else               dist = ctz64(pawns) >> 3;
    }

    for (int i = 0; i < 5; i++) {
        while (dicePiece[pos.dice][i] &&
            (pos.color[pos.side] & pos.piece[i]) == 0 &&
            dist > dicePiece[pos.dice][0]) {
            pos.dice = newDice[pos.dice][i];
        }
    }

    pos.key ^= ZDice[pos.dice];
}

void genLegal(Position& pos, const array<uint64_t, 4>& path, const array<int, 64>& mask, MoveList& ml, int& term) {
#define full(pos,move) uint64_t t=bit(move>>6&63);pos.dice<=6||(pos.dice==24&&pos.color[pos.side]&t)||pos.piece[5]&t
#define zero dice[i]=0;min=0;
    array<int, 255> dice;
    int min = 3;
    genMoves(pos, path, ml);
    term = 0;
    for (int i = 0; i < ml.n; i++) {
        if (full(pos, ml.m[i])) {
            zero
                if (pos.piece[5] & t) {
                    ml.m[0] = ml.m[i];
                    ml.n = 1;
                    term = 1;
                    return;
                }
            continue;
        }
        dice[i] = 1;
        Position pos1 = pos;
        makeMove(pos1, mask, ml.m[i]);
        int move2 = genFirst(pos1, path);
        if (move2 == 0) {
            dice[i] += pos1.dice >= 7;
            if (dice[i] < min)min = dice[i];
            continue;
        }
        if (full(pos1, move2)) {
            zero
                continue;
        }
        Position pos2 = pos1;
        makeMove(pos2, mask, move2);
        if (genFirst(pos2, path)) {
            zero
                continue;
        }
        MoveList ml2;
        genMoves(pos1, path, ml2);
        for (int j = 0; j < ml2.n; j++) {
            if (full(pos1, ml2.m[j])) {
                zero
                    break;
            }
            pos2 = pos1;
            makeMove(pos2, mask, ml2.m[j]);
            if (genFirst(pos2, path)) {
                zero
                    break;
            }
        }
        if (dice[i] < min)min = dice[i];
    }
    for (int i = 0; i < ml.n;)if (dice[i] == min)i++; else {
        ml.n--;
        ml.m[i] = ml.m[ml.n];
        dice[i] = dice[ml.n];
    }
#undef zero
#undef full
}



#include <mutex>


#if defined(_MSC_VER) || defined(__x86_64__) || defined(__i386)
#include <immintrin.h>
#endif


#include <NvInfer.h>
#include <cuda_runtime.h>



static constexpr int NET_BLOCKS = 10;
static constexpr int NET_CHANNELS = 192; // widened from 128 (see widen192 mode)
static constexpr int POLICY_P = 73;
static constexpr double AI_BN_EPS = 1e-5;

// SE (affine)
static constexpr int SE_CHANNELS = 24;   // scaled with C (C/8)

// Heads
static constexpr int HEAD_POLICY_C = 32; // 32 for 10x128 — a standard good choice
static constexpr int HEAD_VALUE_C = 32;
static constexpr int HEAD_VALUE_FC = 256;
static constexpr int POLICY_SIZE = 8 * 8 * POLICY_P; // 4672
static constexpr int POLICY_CHW = POLICY_P * 64;     // 4672 in channel-major [pl][sq]
static constexpr int TRT_MAX_BATCH = 256;              // profile max batch (and fixed target)

// Fast-gather (copy only legal-move logits instead of full 73*64)
static constexpr int AI_MAX_MOVES = 255; // MoveList::m[255]

// NVCC-only CUDA kernels (optional). If you compile as .cu with nvcc -> enabled.
#ifndef AI_HAVE_CUDA_KERNELS
#define AI_HAVE_CUDA_KERNELS 0
#endif
static AI_FORCEINLINE void swapPlane64(float* base, int pA, int pB) {
    float* a = base + pA * 64;
    float* b = base + pB * 64;
    for (int i = 0; i < 64; ++i) std::swap(a[i], b[i]);
}

// x: pointer to ONE position input, layout [plane][sq], planes=NN_SQ_PLANES, sq=0..63 (a1..h8)

static AI_FORCEINLINE bool onBoard(int r, int f) {
    return (unsigned)r < 8u && (unsigned)f < 8u;
}

static AI_FORCEINLINE int makeMoveEnc(int from, int to, int promo) {
    return from | (to << 6) | (promo << 12);
}

static int decodePolicyIndexToMoveCHW(int k) {
    const int plane = k / 64;
    const int from = k - plane * 64;

    const int fr = from >> 3;
    const int ff = from & 7;

    // promo planes: 64..72 (underpromos only)
    if (plane >= 64) {
        const int t = plane - 64;      // 0..8
        const int pGroup = t / 3;      // 0..2
        const int dir3 = t % 3;      // 0..2 -> df = -1,0,+1

        // pGroup mapping must match policyIndexCHW:
        // pGroup = 0 => promo=3 (rook)
        // pGroup = 1 => promo=2 (bishop)
        // pGroup = 2 => promo=1 (knight)
        const int promo = 3 - pGroup;
        const int df = dir3 - 1;
        const int tr = fr + 1;         // forward 1 (us-perspective)
        const int tf = ff + df;

        if (!onBoard(tr, tf)) return makeMoveEnc(from, from, 0);
        const int to = tr * 8 + tf;
        return makeMoveEnc(from, to, promo);
    }

    // knight planes: 56..63
    if (plane >= 56) {
        static constexpr int KNR[8] = { +2, +1, -1, -2, -2, -1, +1, +2 };
        static constexpr int KNF[8] = { +1, +2, +2, +1, -1, -2, -2, -1 };

        const int ki = plane - 56;
        const int tr = fr + KNR[ki];
        const int tf = ff + KNF[ki];

        if (!onBoard(tr, tf)) return makeMoveEnc(from, from, 0);
        const int to = tr * 8 + tf;
        return makeMoveEnc(from, to, 0);
    }

    // sliding planes: 0..55 => dir*7 + (dist-1)
    const int dir = plane / 7;      // 0..7
    const int dist = (plane % 7) + 1;

    int dr = 0, df = 0;
    switch (dir) {
    case 0: dr = +dist; df = 0;      break; // N
    case 1: dr = +dist; df = +dist;  break; // NE
    case 2: dr = 0;     df = +dist;  break; // E
    case 3: dr = -dist; df = +dist;  break; // SE
    case 4: dr = -dist; df = 0;      break; // S
    case 5: dr = -dist; df = -dist;  break; // SW
    case 6: dr = 0;     df = -dist;  break; // W
    case 7: dr = +dist; df = -dist;  break; // NW
    }

    const int tr = fr + dr;
    const int tf = ff + df;

    if (!onBoard(tr, tf)) return makeMoveEnc(from, from, 0);
    const int to = tr * 8 + tf;
    return makeMoveEnc(from, to, 0);
}

static AI_FORCEINLINE int mirrorMoveFile(int move) {
    int from = move & 63;
    int to = (move >> 6) & 63;
    int promo = (move >> 12) & 7;

    from ^= 7;
    to ^= 7;
    return makeMoveEnc(from, to, promo);
}





static AI_FORCEINLINE bool fileExists(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return (bool)f;
}

static void readFileAll(const std::string& path, std::vector<char>& out) {
    out.clear();
    std::ifstream is(path, std::ios::binary | std::ios::ate);
    if (!is) return;
    std::streamsize n = is.tellg();
    if (n <= 0) return;
    out.resize((size_t)n);
    is.seekg(0, std::ios::beg);
    is.read(out.data(), n);
}

static bool writeFileAll(const std::string& path, const void* data, size_t size) {
    std::ofstream os(path, std::ios::binary);
    if (!os) return false;
    os.write((const char*)data, (std::streamsize)size);
    return (bool)os;
}

static AI_FORCEINLINE size_t volume(const nvinfer1::Dims& d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) v *= (size_t)d.d[i];
    return v;
}

static void softmaxLocal(std::vector<float>& x) {
    if (x.empty()) return;
    float mx = x[0];
    for (float v : x) if (v > mx) mx = v;
    double sum = 0.0;
    for (float& v : x) { v = std::exp(v - mx); sum += v; }
    if (!(sum > 0.0)) {
        float inv = 1.0f / (float)x.size();
        for (float& v : x) v = inv;
        return;
    }
    float inv = (float)(1.0 / sum);
    for (float& v : x) v *= inv;
}

static AI_FORCEINLINE float clamp01(float v) {
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
}

static AI_FORCEINLINE uint16_t quantizeProbU16(float p) {
    if (!(p > 0.0f)) return 0u;
    if (p >= 1.0f) return 65535u;
    return (uint16_t)lrintf(p * 65535.0f);
}

static AI_FORCEINLINE float dequantizeProbU16(uint16_t q) {
    return (float)q * (1.0f / 65535.0f);
}


static std::mutex g_diagMutex;
static std::ofstream g_diagFile;

static std::string diagNowStr() {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto tt = system_clock::to_time_t(now);

    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif

    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);

    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

    std::ostringstream oss;
    oss << buf << "." << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

static void diagInit(const std::string& path = "crash.log") {
    std::lock_guard<std::mutex> lk(g_diagMutex);
    if (!g_diagFile.is_open()) {
        g_diagFile.open(path, std::ios::out | std::ios::app);
        g_diagFile.setf(std::ios::unitbuf);
    }
}

static void diagLogLine(const std::string& msg) {
    std::lock_guard<std::mutex> lk(g_diagMutex);

    std::ostringstream line;
    line << "[" << diagNowStr() << "][tid " << std::this_thread::get_id() << "] " << msg;


    if (g_diagFile.is_open()) {
        g_diagFile << line.str() << std::endl;
    }
}

static void onTerminateHandler() noexcept {
    try {
        auto ep = std::current_exception();
        if (ep) std::rethrow_exception(ep);
        diagLogLine("[terminate] called with no active exception");
    }
    catch (const std::exception& e) {
        diagLogLine(std::string("[terminate] std::exception: ") + e.what());
    }
    catch (...) {
        diagLogLine("[terminate] unknown exception");
    }

    std::abort();
}

static void onSignalHandler(int sig) {
    std::ostringstream oss;
    oss << "[signal] received signal " << sig;
    diagLogLine(oss.str());

    std::_Exit(128 + sig);
}

#if defined(_WIN32)
static LONG WINAPI topLevelExceptionFilter(EXCEPTION_POINTERS* ep) {
    if (!ep || !ep->ExceptionRecord) {
        diagLogLine("[SEH] unhandled Windows exception (no details)");
        return EXCEPTION_EXECUTE_HANDLER;
    }

    std::ostringstream oss;
    oss << "[SEH] unhandled exception code=0x"
        << std::hex << std::uppercase
        << (unsigned long)ep->ExceptionRecord->ExceptionCode
        << " address=" << ep->ExceptionRecord->ExceptionAddress;
    diagLogLine(oss.str());

    return EXCEPTION_EXECUTE_HANDLER;
}
#endif

static void installCrashDiagnostics() {
    diagInit("crash.log");
    std::set_terminate(onTerminateHandler);

    std::signal(SIGABRT, onSignalHandler);
    std::signal(SIGSEGV, onSignalHandler);
    std::signal(SIGILL, onSignalHandler);
    std::signal(SIGFPE, onSignalHandler);

#if defined(_WIN32)
    SetUnhandledExceptionFilter(topLevelExceptionFilter);
#endif

    diagLogLine("[diag] crash diagnostics installed");
}

static void cudaCheck(cudaError_t e, const char* expr, const char* file, int line) {
    if (e == cudaSuccess) return;

    std::ostringstream oss;
    oss << "[CUDA FATAL] "
        << cudaGetErrorName(e) << ": "
        << cudaGetErrorString(e)
        << " at " << file << ":" << line
        << " in " << expr;

    diagLogLine(oss.str());
    std::abort();
}
#define CUDA_CHECK(x) cudaCheck((x), #x, __FILE__, __LINE__)



struct TrtLogger final : nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TRT] " << msg << "\n";
        }
    }
};

static TrtLogger g_trtLogger;





// ============================================================
// Weight store for building TRT network
// ============================================================

struct WeightStore {
    std::vector<std::vector<float>> fBlocks;
    std::vector<std::vector<int32_t>> iBlocks;

    nvinfer1::Weights add(std::vector<float>&& v) {
        fBlocks.push_back(std::move(v));
        auto& b = fBlocks.back();
        nvinfer1::Weights w{};
        w.type = nvinfer1::DataType::kFLOAT;
        w.values = b.data();
        w.count = (int64_t)b.size();
        return w;
    }

    nvinfer1::Weights addI32(std::vector<int32_t>&& v) {
        iBlocks.push_back(std::move(v));
        auto& b = iBlocks.back();
        nvinfer1::Weights w{};
        w.type = nvinfer1::DataType::kINT32;
        w.values = b.data();
        w.count = (int64_t)b.size();
        return w;
    }

    nvinfer1::Weights addZeros(size_t n) {
        std::vector<float> v(n, 0.0f);
        return add(std::move(v));
    }
};

static AI_FORCEINLINE nvinfer1::Dims dims1(int n) {
    nvinfer1::Dims d{};
    d.nbDims = 1;
    d.d[0] = n;
    return d;
}

static void fillHe(std::vector<float>& w, int fanIn, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.0f, std::sqrt(2.0f / (float)fanIn));
    for (float& x : w) x = nd(rng);
}

static void fillSmall(std::vector<float>& w, float stddev, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.0f, stddev);
    for (float& x : w) x = nd(rng);
}

// ========================= BN+SE ResNet builder for TensorRT =========================






static nvinfer1::ITensor* addRelu(nvinfer1::INetworkDefinition& net, nvinfer1::ITensor& x) {
    auto* a = net.addActivation(x, nvinfer1::ActivationType::kRELU);
    if (!a) return nullptr;
    return a->getOutput(0);
}
static nvinfer1::ITensor* addSigmoid(nvinfer1::INetworkDefinition& net, nvinfer1::ITensor& x) {
    auto* a = net.addActivation(x, nvinfer1::ActivationType::kSIGMOID);
    if (!a) return nullptr;
    return a->getOutput(0);
}
static nvinfer1::ITensor* addSum(nvinfer1::INetworkDefinition& net,
    nvinfer1::ITensor& a,
    nvinfer1::ITensor& b) {
    auto* e = net.addElementWise(a, b, nvinfer1::ElementWiseOperation::kSUM);
    if (!e) return nullptr;
    return e->getOutput(0);
}
static nvinfer1::ITensor* addProd(nvinfer1::INetworkDefinition& net,
    nvinfer1::ITensor& a,
    nvinfer1::ITensor& b) {
    auto* e = net.addElementWise(a, b, nvinfer1::ElementWiseOperation::kPROD);
    if (!e) return nullptr;
    return e->getOutput(0);
}

static nvinfer1::IConvolutionLayer* addConv3x3NoBiasNamed(nvinfer1::INetworkDefinition& net,
    nvinfer1::ITensor& x,
    int outC,
    const nvinfer1::Weights& W,
    const char* name) {
    nvinfer1::Weights noBias{};
    auto* c = net.addConvolutionNd(x, outC, nvinfer1::DimsHW{ 3,3 }, W, noBias);
    if (!c) return nullptr;
    c->setStrideNd(nvinfer1::DimsHW{ 1,1 });
    c->setPaddingNd(nvinfer1::DimsHW{ 1,1 });
    c->setName(name);
    return c;
}
static nvinfer1::IConvolutionLayer* addConv1x1NoBiasNamed(nvinfer1::INetworkDefinition& net,
    nvinfer1::ITensor& x,
    int outC,
    const nvinfer1::Weights& W,
    const char* name) {
    nvinfer1::Weights noBias{};
    auto* c = net.addConvolutionNd(x, outC, nvinfer1::DimsHW{ 1,1 }, W, noBias);
    if (!c) return nullptr;
    c->setStrideNd(nvinfer1::DimsHW{ 1,1 });
    c->setPaddingNd(nvinfer1::DimsHW{ 0,0 });
    c->setName(name);
    return c;
}
static nvinfer1::IConvolutionLayer* addConv1x1WithBiasNamed(nvinfer1::INetworkDefinition& net,
    nvinfer1::ITensor& x,
    int outC,
    const nvinfer1::Weights& W,
    const nvinfer1::Weights& B,
    const char* name) {
    auto* c = net.addConvolutionNd(x, outC, nvinfer1::DimsHW{ 1,1 }, W, B);
    if (!c) return nullptr;
    c->setStrideNd(nvinfer1::DimsHW{ 1,1 });
    c->setPaddingNd(nvinfer1::DimsHW{ 0,0 });
    c->setName(name);
    return c;
}

// BN-inference as Scale: y = x*scale + shift
// Refittable: layerName with roles kSCALE/kSHIFT.
static nvinfer1::ITensor* addBatchNorm2dScaleNamed(nvinfer1::INetworkDefinition& net,
    WeightStore& store,
    nvinfer1::ITensor& x,
    int C,
    const char* name) {
    std::vector<float> scale((size_t)C, 1.0f);
    std::vector<float> shift((size_t)C, 0.0f);

    auto Wscale = store.add(std::move(scale));
    auto Wshift = store.add(std::move(shift));
    nvinfer1::Weights Wpower{}; // empty => power=1

    auto* s = net.addScaleNd(x, nvinfer1::ScaleMode::kCHANNEL, Wshift, Wscale, Wpower, /*channelAxis=*/1);
    if (!s) return nullptr;
    s->setName(name);
    return s->getOutput(0);
}

static nvinfer1::ITensor* addSEBlockAffineNamed(nvinfer1::INetworkDefinition& net,
    WeightStore& store,
    std::mt19937& rng,
    nvinfer1::ITensor& x,
    int C,
    int seC,
    const std::string& prefix) {
    using namespace nvinfer1;

    // GAP: [N,C,8,8] -> [N,C,1,1]
    auto* red = net.addReduce(x, ReduceOperation::kAVG, (1u << 2) | (1u << 3), /*keepDims=*/true);
    if (!red) return nullptr;
    red->setName((prefix + ".se.pool").c_str());
    ITensor* s = red->getOutput(0);

    // fc1: C -> seC  (bias=true)
    {
        std::vector<float> w((size_t)seC * (size_t)C);
        std::vector<float> b((size_t)seC, 0.0f);
        fillSmall(w, 1e-2f, rng);

        auto W = store.add(std::move(w));
        auto B = store.add(std::move(b));

        auto* c = addConv1x1WithBiasNamed(net, *s, seC, W, B, (prefix + ".se.fc1").c_str());
        if (!c) return nullptr;

        s = addRelu(net, *c->getOutput(0));
        if (!s) return nullptr;
    }

    // fc2: seC -> 2C (bias=true), output [N,2C,1,1]
    ITensor* s2 = nullptr;
    {
        std::vector<float> w((size_t)(2 * C) * (size_t)seC);
        std::vector<float> b((size_t)(2 * C), 0.0f);
        fillSmall(w, 1e-2f, rng);

        auto W = store.add(std::move(w));
        auto B = store.add(std::move(b));

        auto* c = addConv1x1WithBiasNamed(net, *s, 2 * C, W, B, (prefix + ".se.fc2").c_str());
        if (!c) return nullptr;

        s2 = c->getOutput(0);
    }

    std::vector<int32_t> idxW((size_t)C), idxB((size_t)C);
    for (int i = 0; i < C; ++i) {
        idxW[(size_t)i] = i;
        idxB[(size_t)i] = C + i;
    }

    auto Widx = store.addI32(std::move(idxW));
    auto Bidx = store.addI32(std::move(idxB));

    auto* cW = net.addConstant(dims1(C), Widx);
    auto* cB = net.addConstant(dims1(C), Bidx);
    if (!cW || !cB) return nullptr;

    cW->setName((prefix + ".se.idxW").c_str());
    cB->setName((prefix + ".se.idxB").c_str());

    auto* gW = net.addGather(*s2, *cW->getOutput(0), 1);
    auto* gB = net.addGather(*s2, *cB->getOutput(0), 1);
    if (!gW || !gB) return nullptr;

    gW->setName((prefix + ".se.gatherW").c_str());
    gB->setName((prefix + ".se.gatherB").c_str());

    ITensor* Wt = gW->getOutput(0); // [N,C,1,1]
    ITensor* Bt = gB->getOutput(0); // [N,C,1,1]

    ITensor* Z = addSigmoid(net, *Wt); // [N,C,1,1]
    if (!Z) return nullptr;

    // out = sigmoid(W) * x + B
    ITensor* y = addProd(net, x, *Z);
    if (!y) return nullptr;

    ITensor* out = addSum(net, *y, *Bt);
    if (!out) return nullptr;

    return out;
}

static nvinfer1::ITensor* addResBlockSEBNNamed(nvinfer1::INetworkDefinition& net,
    WeightStore& store,
    std::mt19937& rng,
    nvinfer1::ITensor& xIn,
    int C,
    int bi) {
    using namespace nvinfer1;

    ITensor* skip = &xIn;
    ITensor* x = &xIn;

    // conv1 -> bn1 -> relu
    {
        std::vector<float> w((size_t)C * (size_t)C * 9u);
        fillHe(w, C * 3 * 3, rng);
        auto W = store.add(std::move(w));

        std::string nConv = "block" + std::to_string(bi) + ".conv1";
        auto* c1 = addConv3x3NoBiasNamed(net, *x, C, W, nConv.c_str());
        if (!c1) return nullptr;
        x = c1->getOutput(0);

        std::string nBN = "block" + std::to_string(bi) + ".bn1";
        x = addBatchNorm2dScaleNamed(net, store, *x, C, nBN.c_str());
        if (!x) return nullptr;

        x = addRelu(net, *x);
        if (!x) return nullptr;
    }

    // conv2 -> bn2
    {
        std::vector<float> w((size_t)C * (size_t)C * 9u);
        fillHe(w, C * 3 * 3, rng);
        auto W = store.add(std::move(w));

        std::string nConv = "block" + std::to_string(bi) + ".conv2";
        auto* c2 = addConv3x3NoBiasNamed(net, *x, C, W, nConv.c_str());
        if (!c2) return nullptr;
        x = c2->getOutput(0);

        std::string nBN = "block" + std::to_string(bi) + ".bn2";
        x = addBatchNorm2dScaleNamed(net, store, *x, C, nBN.c_str());
        if (!x) return nullptr;
    }

    // SE affine
    x = addSEBlockAffineNamed(net, store, rng, *x, C, SE_CHANNELS, "block" + std::to_string(bi));
    if (!x) return nullptr;

    // add + relu
    x = addSum(net, *x, *skip);
    if (!x) return nullptr;

    x = addRelu(net, *x);
    if (!x) return nullptr;

    return x;
}

static bool buildAndSavePlan(const std::string& planFile) {
    using namespace nvinfer1;

    std::unique_ptr<IBuilder> builder(createInferBuilder(g_trtLogger));
    if (!builder) return false;

    std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
    if (!config) return false;

    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1ull << 30);
    config->setMaxAuxStreams(7);

    if (builder->platformHasFastFp16()) config->setFlag(BuilderFlag::kFP16);

    // Make precision/type constraints predictable when we force FP32 outputs.
#if defined(NV_TENSORRT_MAJOR)
    // Flag exists in TRT8+. If you compile with older TRT, just remove this block.
    config->setFlag(BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);
#endif

    // refit-ready
    config->setFlag(BuilderFlag::kREFIT);

    const uint32_t flags = 1u << (uint32_t)NetworkDefinitionCreationFlag::kEXPLICIT_BATCH;
    std::unique_ptr<INetworkDefinition> net(builder->createNetworkV2(flags));
    if (!net) return false;

    ITensor* in = net->addInput("input", DataType::kFLOAT, Dims4{ -1, NN_SQ_PLANES, 8, 8 });
    if (!in) return false;

    IOptimizationProfile* prof = builder->createOptimizationProfile();
    if (!prof) return false;

    prof->setDimensions("input", OptProfileSelector::kMIN, Dims4{ 1, NN_SQ_PLANES, 8, 8 });
    prof->setDimensions("input", OptProfileSelector::kOPT, Dims4{ 64, NN_SQ_PLANES, 8, 8 });
    prof->setDimensions("input", OptProfileSelector::kMAX, Dims4{ TRT_MAX_BATCH, NN_SQ_PLANES, 8, 8 });
    if (!prof->isValid()) return false;
    if (config->addOptimizationProfile(prof) < 0) return false;

    std::mt19937 rng(0x12345678u);
    WeightStore store;

    // =========================
    // Stem: conv3x3(no bias) -> bn -> relu
    // =========================
    ITensor* x = nullptr;
    {
        std::vector<float> w((size_t)NET_CHANNELS * (size_t)NN_SQ_PLANES * 9u);
        fillHe(w, NN_SQ_PLANES * 3 * 3, rng);
        auto W = store.add(std::move(w));

        auto* stem = addConv3x3NoBiasNamed(*net, *in, NET_CHANNELS, W, "stem.conv");
        if (!stem) return false;
        x = stem->getOutput(0);

        x = addBatchNorm2dScaleNamed(*net, store, *x, NET_CHANNELS, "stem.bn");
        if (!x) return false;

        x = addRelu(*net, *x);
        if (!x) return false;
    }

    // =========================
    // 10 residual blocks (BN + Affine-SE)
    // =========================
    for (int bi = 0; bi < NET_BLOCKS; ++bi) {
        x = addResBlockSEBNNamed(*net, store, rng, *x, NET_CHANNELS, bi);
        if (!x) return false;
    }

    // =========================
    // Policy head: 1x1(no bias)->bn->relu->1x1(with bias)->logits
    // OUTPUT MUST BE FP32
    // =========================
    {
        // conv1 C->HEAD_POLICY_C
        std::vector<float> w1((size_t)HEAD_POLICY_C * (size_t)NET_CHANNELS);
        fillHe(w1, NET_CHANNELS, rng);
        auto W1 = store.add(std::move(w1));

        auto* c1 = addConv1x1NoBiasNamed(*net, *x, HEAD_POLICY_C, W1, "head.policy.conv1");
        if (!c1) return false;
        ITensor* p = c1->getOutput(0);

        p = addBatchNorm2dScaleNamed(*net, store, *p, HEAD_POLICY_C, "head.policy.bn1");
        if (!p) return false;

        p = addRelu(*net, *p);
        if (!p) return false;

        // conv2 -> 73 (with bias)
        std::vector<float> w2((size_t)POLICY_P * (size_t)HEAD_POLICY_C);
        std::vector<float> b2((size_t)POLICY_P, 0.0f);
        fillSmall(w2, 1e-3f, rng);

        auto W2 = store.add(std::move(w2));
        auto B2 = store.add(std::move(b2));

        auto* c2 = addConv1x1WithBiasNamed(*net, *p, POLICY_P, W2, B2, "head.policy.conv2");
        if (!c2) return false;

        ITensor* polRaw = c2->getOutput(0);
        // Force output binding dtype to FP32
        polRaw->setType(DataType::kFLOAT);

        // Extra-robust: explicit identity cast at the very end
        auto* polId = net->addIdentity(*polRaw);
        if (!polId) return false;
        polId->setName("policy.cast");
        polId->setOutputType(0, DataType::kFLOAT);

        ITensor* polOut = polId->getOutput(0);
        polOut->setType(DataType::kFLOAT);
        polOut->setName("policy");
        net->markOutput(*polOut);
    }

    // =========================
    // Value head: 1x1(no bias)->bn->relu->flatten->FC->relu->FC->sigmoid
    // OUTPUT MUST BE FP32
    // =========================
    {
        // conv1 C->HEAD_VALUE_C
        std::vector<float> w1((size_t)HEAD_VALUE_C * (size_t)NET_CHANNELS);
        fillHe(w1, NET_CHANNELS, rng);
        auto W1 = store.add(std::move(w1));

        auto* c1 = addConv1x1NoBiasNamed(*net, *x, HEAD_VALUE_C, W1, "head.value.conv1");
        if (!c1) return false;
        ITensor* v = c1->getOutput(0);

        v = addBatchNorm2dScaleNamed(*net, store, *v, HEAD_VALUE_C, "head.value.bn1");
        if (!v) return false;

        v = addRelu(*net, *v);
        if (!v) return false;

        // flatten [B, HEAD_VALUE_C*64]
        auto* sh = net->addShuffle(*v);
        if (!sh) return false;
        sh->setReshapeDimensions(Dims2{ -1, HEAD_VALUE_C * 64 });
        sh->setName("head.value.flatten");
        ITensor* v2d = sh->getOutput(0);
        if (!v2d) return false;

        // FC1 constant [in,out]
        std::vector<float> wFC1((size_t)(HEAD_VALUE_C * 64) * (size_t)HEAD_VALUE_FC);
        fillHe(wFC1, HEAD_VALUE_C * 64, rng);
        auto WFC1 = store.add(std::move(wFC1));
        auto* fc1W = net->addConstant(Dims2{ HEAD_VALUE_C * 64, HEAD_VALUE_FC }, WFC1);
        if (!fc1W) return false;
        fc1W->setName("head.value.fc1.w");

        auto* mm1 = net->addMatrixMultiply(*v2d, MatrixOperation::kNONE,
            *fc1W->getOutput(0), MatrixOperation::kNONE);
        if (!mm1) return false;
        mm1->setName("head.value.fc1.mm");
        ITensor* h1 = mm1->getOutput(0);

        // FC1 bias [1,out]
        {
            std::vector<float> b1((size_t)HEAD_VALUE_FC, 0.0f);
            auto B1 = store.add(std::move(b1));
            auto* cb = net->addConstant(Dims2{ 1, HEAD_VALUE_FC }, B1);
            if (!cb) return false;
            cb->setName("head.value.fc1.b");

            auto* add = net->addElementWise(*h1, *cb->getOutput(0), ElementWiseOperation::kSUM);
            if (!add) return false;
            add->setName("head.value.fc1.addbias");
            h1 = add->getOutput(0);
        }

        auto* rel1 = net->addActivation(*h1, ActivationType::kRELU);
        if (!rel1) return false;
        rel1->setName("head.value.fc1.relu");
        ITensor* h1r = rel1->getOutput(0);

        // FC2 constant [out=1]
        std::vector<float> wFC2((size_t)HEAD_VALUE_FC);
        fillSmall(wFC2, 1e-3f, rng);
        auto WFC2 = store.add(std::move(wFC2));
        auto* fc2W = net->addConstant(Dims2{ HEAD_VALUE_FC, 1 }, WFC2);
        if (!fc2W) return false;
        fc2W->setName("head.value.fc2.w");

        auto* mm2 = net->addMatrixMultiply(*h1r, MatrixOperation::kNONE,
            *fc2W->getOutput(0), MatrixOperation::kNONE);
        if (!mm2) return false;
        mm2->setName("head.value.fc2.mm");
        ITensor* h2 = mm2->getOutput(0);

        // FC2 bias [1,1]
        {
            std::vector<float> b2(1u, 0.0f);
            auto B2 = store.add(std::move(b2));
            auto* cb = net->addConstant(Dims2{ 1, 1 }, B2);
            if (!cb) return false;
            cb->setName("head.value.fc2.b");

            auto* add = net->addElementWise(*h2, *cb->getOutput(0), ElementWiseOperation::kSUM);
            if (!add) return false;
            add->setName("head.value.fc2.addbias");
            h2 = add->getOutput(0);
        }

        auto* sig = net->addActivation(*h2, ActivationType::kSIGMOID);
        if (!sig) return false;
        sig->setName("head.value.sigmoid");
        ITensor* valRaw = sig->getOutput(0);

        // Force output binding dtype to FP32
        valRaw->setType(DataType::kFLOAT);

        // Extra-robust: explicit identity cast at the very end
        auto* valId = net->addIdentity(*valRaw);
        if (!valId) return false;
        valId->setName("value.cast");
        valId->setOutputType(0, DataType::kFLOAT);

        ITensor* valOut = valId->getOutput(0);
        valOut->setType(DataType::kFLOAT);
        valOut->setName("value");
        net->markOutput(*valOut);
    }

    IHostMemory* plan = builder->buildSerializedNetwork(*net, *config);
    if (!plan) return false;

    bool ok = writeFileAll(planFile, plan->data(), (size_t)plan->size());
    delete plan;
    return ok;
}

// ============================================================
// Forward decl (used by TrtRunner::inferBatchGather definition later)
// ============================================================

struct PendingNN;

// ============================================================
// TensorRT runtime inference wrapper
// - Fixed batch=256
// - CUDA Graph capture
// - Optional GPU gather for legal moves (cuts D2H from 4672 floats/pos to <=255 floats/pos)
// ============================================================
extern "C" void launchGatherPolicyKernel(const float* policy,
    const int* idx,
    float* out,
    int total,
    cudaStream_t stream);
struct TrtRunner {
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* ctx = nullptr;

    cudaStream_t stream = nullptr;

    void* dInput = nullptr;  // [256,25,8,8] float
    void* dPolicy = nullptr;  // [256,73,8,8] float
    void* dValue = nullptr;  // [256,1] float
    // Aux streams for TensorRT (needed for stable CUDA Graph capture when engine uses aux streams)
    std::vector<cudaStream_t> auxStreams;
    int nbAuxStreams = 0;
#if AI_HAVE_CUDA_KERNELS
    void* dGatherIdx = nullptr; // [256,AI_MAX_MOVES] int32
    void* dGatherLogits = nullptr; // [256,AI_MAX_MOVES] float
#endif

    int maxBatch = TRT_MAX_BATCH;
    int currentShapeB = -1;

    // Pinned host buffers
    float* hInputPinned = nullptr; // 256 * 1600

    // Full-policy pinned (kept for debug / compatibility)
    float* hPolicyPinned = nullptr; // 256 * 4672 (CHW)
    float* hValuePinned = nullptr; // 256

#if AI_HAVE_CUDA_KERNELS
    int* hGatherIdxPinned = nullptr; // 256 * AI_MAX_MOVES
    float* hGatherLogitsPinned = nullptr; // 256 * AI_MAX_MOVES
#endif

    // CUDA Graph
    bool graphReady = false;
    cudaGraph_t     graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;

    AI_FORCEINLINE size_t inBytes(int B) const {
        return (size_t)B * (size_t)NN_INPUT_SIZE * sizeof(float);
    }
    AI_FORCEINLINE size_t polBytes(int B) const {
        return (size_t)B * (size_t)POLICY_SIZE * sizeof(float);
    }
    AI_FORCEINLINE size_t valBytes(int B) const {
        return (size_t)B * sizeof(float);
    }

#if AI_HAVE_CUDA_KERNELS
    AI_FORCEINLINE size_t gatherIdxBytes(int B) const {
        return (size_t)B * (size_t)AI_MAX_MOVES * sizeof(int);
    }
    AI_FORCEINLINE size_t gatherLogitsBytes(int B) const {
        return (size_t)B * (size_t)AI_MAX_MOVES * sizeof(float);
    }
#endif

    AI_FORCEINLINE size_t inBytesFull() const { return inBytes(maxBatch); }
    AI_FORCEINLINE size_t polBytesFull() const { return polBytes(maxBatch); }
    AI_FORCEINLINE size_t valBytesFull() const { return valBytes(maxBatch); }
#if AI_HAVE_CUDA_KERNELS
    AI_FORCEINLINE size_t gatherIdxBytesFull() const { return gatherIdxBytes(maxBatch); }
    AI_FORCEINLINE size_t gatherLogitsBytesFull() const { return gatherLogitsBytes(maxBatch); }
#endif

    bool ensureShape(int B) {
        if (!ctx) return false;
        if (currentShapeB == B) return true;

        if (!ctx->setInputShape("input", nvinfer1::Dims4{ B, NN_SQ_PLANES, 8, 8 })) {
            std::cerr << "TensorRT: setInputShape(" << B << ",25,8,8) failed.\n";
            return false;
        }
        currentShapeB = B;
        return true;
    }

    // Host accessors
    AI_FORCEINLINE const float* policyHostPtr(int i) const {
        return hPolicyPinned + (size_t)i * (size_t)POLICY_SIZE;
    }
    AI_FORCEINLINE float valueHost(int i) const {
        return hValuePinned[(size_t)i];
    }
#if AI_HAVE_CUDA_KERNELS
    AI_FORCEINLINE const float* gatherLogitsHostPtr(int i) const {
        return hGatherLogitsPinned + (size_t)i * (size_t)AI_MAX_MOVES;
    }
#endif

    void copyValuesTo(float* outValue, int B) const {
        if (!outValue || B <= 0) return;
        std::memcpy(outValue, hValuePinned, valBytes(B));
    }

#if AI_HAVE_CUDA_KERNELS
    void copyGatherLogitsTo(float* outLogits, int B) const {
        if (!outLogits || B <= 0) return;
        std::memcpy(outLogits, hGatherLogitsPinned, gatherLogitsBytes(B));
    }
#endif

    void copyPolicyTo(float* outPolicy, int B) const {
        if (!outPolicy || B <= 0) return;
        std::memcpy(outPolicy, hPolicyPinned,
            (size_t)B * (size_t)POLICY_SIZE * sizeof(float));
    }
    bool setupAuxStreams() {
        if (!engine || !ctx) return false;

        nbAuxStreams = (int)engine->getNbAuxStreams();
        if (nbAuxStreams <= 0) {
            // no aux streams used by engine
            for (cudaStream_t s : auxStreams) if (s) cudaStreamDestroy(s);
            auxStreams.clear();
            nbAuxStreams = 0;
            return true;
        }

        // destroy old aux streams (if any)
        for (cudaStream_t s : auxStreams) {
            if (s) cudaStreamDestroy(s);
        }
        auxStreams.clear();
        auxStreams.resize((size_t)nbAuxStreams, nullptr);

        int leastPrio = 0, greatestPrio = 0;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&leastPrio, &greatestPrio));

        // create non-default, non-blocking streams (important for CUDA graph capture stability)
        for (int i = 0; i < nbAuxStreams; ++i) {
            CUDA_CHECK(cudaStreamCreateWithPriority(&auxStreams[(size_t)i],
                cudaStreamNonBlocking,
                greatestPrio));
        }

        // IMPORTANT: setAuxStreams() is void in TRT API (no return value).
        // Must not pass default stream, and streams must be distinct.
        ctx->setAuxStreams(auxStreams.data(), (int32_t)auxStreams.size());
        return true;
    }
    bool captureCudaGraphFixed256() {
        if (!ctx || !stream) return false;
        if (!ensureShape(TRT_MAX_BATCH)) return false;

        // Ensure aux streams are attached before warmup/capture
        if (nbAuxStreams > 0 && (int)auxStreams.size() == nbAuxStreams) {
            ctx->setAuxStreams(auxStreams.data(), (int32_t)auxStreams.size());
        }

        // Warm up once
        CUDA_CHECK(cudaMemsetAsync(dInput, 0, inBytesFull(), stream));
#if AI_HAVE_CUDA_KERNELS
        CUDA_CHECK(cudaMemsetAsync(dGatherIdx, 0, gatherIdxBytesFull(), stream));
        CUDA_CHECK(cudaMemsetAsync(dGatherLogits, 0, gatherLogitsBytesFull(), stream));
#endif
        if (!ctx->enqueueV3(stream)) return false;
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::memset(hInputPinned, 0, inBytesFull());
        std::memset(hValuePinned, 0, valBytesFull());
#if AI_HAVE_CUDA_KERNELS
        std::fill_n(hGatherIdxPinned, (size_t)TRT_MAX_BATCH * (size_t)AI_MAX_MOVES, -1);
        std::memset(hGatherLogitsPinned, 0, gatherLogitsBytesFull());
#endif

        // IMPORTANT: begin capture on a non-blocking stream
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

        // Make sure aux streams are set for the captured enqueue as well
        if (nbAuxStreams > 0 && (int)auxStreams.size() == nbAuxStreams) {
            ctx->setAuxStreams(auxStreams.data(), (int32_t)auxStreams.size());
        }

        // H2D input
        CUDA_CHECK(cudaMemcpyAsync(dInput, hInputPinned, inBytesFull(),
            cudaMemcpyHostToDevice, stream));

#if AI_HAVE_CUDA_KERNELS
        // H2D gather indices
        CUDA_CHECK(cudaMemcpyAsync(dGatherIdx, hGatherIdxPinned, gatherIdxBytesFull(),
            cudaMemcpyHostToDevice, stream));
#endif

        if (!ctx->enqueueV3(stream)) {
            CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
            graph = nullptr;
            return false;
        }

#if AI_HAVE_CUDA_KERNELS
        // Gather kernel: policy -> logits per move
        {
            const int total = TRT_MAX_BATCH * AI_MAX_MOVES;
            launchGatherPolicyKernel((const float*)dPolicy,
                (const int*)dGatherIdx,
                (float*)dGatherLogits,
                total,
                stream);

            // optional during debugging:
            CUDA_CHECK(cudaGetLastError());
        }

        // D2H only gathered logits + value
        CUDA_CHECK(cudaMemcpyAsync(hGatherLogitsPinned, dGatherLogits, gatherLogitsBytesFull(),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(hValuePinned, dValue, valBytesFull(),
            cudaMemcpyDeviceToHost, stream));
#else
        // Fallback: D2H full policy + value
        CUDA_CHECK(cudaMemcpyAsync(hPolicyPinned, dPolicy, polBytesFull(),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(hValuePinned, dValue, valBytesFull(),
            cudaMemcpyDeviceToHost, stream));
#endif

        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        if (!graph) return false;

        cudaError_t e = cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
        if (e != cudaSuccess || !graphExec) {
            cudaGraphDestroy(graph);
            graph = nullptr;
            graphExec = nullptr;
            return false;
        }

        graphReady = true;
        return true;
    }

    bool initFromPlan(const std::string& planFile) {
        std::vector<char> blob;
        readFileAll(planFile, blob);
        if (blob.empty()) return false;

        runtime = nvinfer1::createInferRuntime(g_trtLogger);
        if (!runtime) return false;

        engine = runtime->deserializeCudaEngine(blob.data(), blob.size());
        if (!engine) return false;

        // IMPORTANT: verify output binding dtypes. If they are FP16, our float buffers are invalid.
        {
            auto dtPol = engine->getTensorDataType("policy");
            auto dtVal = engine->getTensorDataType("value");

            if (dtPol != nvinfer1::DataType::kFLOAT || dtVal != nvinfer1::DataType::kFLOAT) {
                std::cerr << "TensorRT plan has non-FP32 outputs: "
                    << "policy=" << (int)dtPol << " value=" << (int)dtVal
                    << ". Delete plan and rebuild.\n";
                return false; // initOrCreate() will rebuild
            }
        }

        ctx = engine->createExecutionContext();
        if (!ctx) return false;

        int leastPrio = 0, greatestPrio = 0;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&leastPrio, &greatestPrio));
        CUDA_CHECK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatestPrio));

        const size_t inBytes = inBytesFull();
        const size_t polBytes = polBytesFull(); // safe because policy is FP32
        const size_t valBytes = valBytesFull(); // safe because value is FP32

        CUDA_CHECK(cudaMalloc(&dInput, inBytes));
        CUDA_CHECK(cudaMalloc(&dPolicy, polBytes));
        CUDA_CHECK(cudaMalloc(&dValue, valBytes));

        CUDA_CHECK(cudaMallocHost((void**)&hInputPinned, inBytes));
        CUDA_CHECK(cudaMallocHost((void**)&hPolicyPinned, polBytes)); // keep for debug
        CUDA_CHECK(cudaMallocHost((void**)&hValuePinned, valBytes));

#if AI_HAVE_CUDA_KERNELS
        CUDA_CHECK(cudaMalloc(&dGatherIdx, gatherIdxBytesFull()));
        CUDA_CHECK(cudaMalloc(&dGatherLogits, gatherLogitsBytesFull()));
        CUDA_CHECK(cudaMallocHost((void**)&hGatherIdxPinned, gatherIdxBytesFull()));
        CUDA_CHECK(cudaMallocHost((void**)&hGatherLogitsPinned, gatherLogitsBytesFull()));
#endif

        // Set IO addresses (names must match markOutput() names)
        if (!ctx->setTensorAddress("policy", dPolicy)) return false;
        if (!ctx->setTensorAddress("value", dValue))  return false;
        if (!ctx->setInputTensorAddress("input", dInput)) return false;

        // Profile 0 (only one)
        if (!ctx->setOptimizationProfileAsync(0, stream)) return false;

        currentShapeB = -1;
        if (!ensureShape(TRT_MAX_BATCH)) {
            std::cerr << "TensorRT: initial setInputShape failed.\n";
            return false;
        }

        // IMPORTANT: attach aux streams BEFORE graph capture (if engine uses them)
        if (!setupAuxStreams()) {
            std::cerr << "TensorRT: setupAuxStreams failed (engine may still run, but capture may fail).\n";
            // continue anyway
        }

        if (!captureCudaGraphFixed256()) {
            std::cerr << "TensorRT: CUDA Graph capture failed; falling back to non-graph path.\n";
            graphReady = false;
        }

        return true;
    }

    bool initOrCreate(const std::string& planFile) {
        if (fileExists(planFile)) {
            if (initFromPlan(planFile)) return true;
            shutdown();
        }

        std::cout << "TensorRT plan file '" << planFile << "' was not found or could not be loaded — building engine...\n";
        if (!buildAndSavePlan(planFile)) {
            std::cerr << "Failed to build and save TensorRT plan '" << planFile << "'.\n";
            return false;
        }
        std::cout << "Built and saved '" << planFile << "'. Loading...\n";

        if (!initFromPlan(planFile)) {
            std::cerr << "Failed to load TensorRT plan after building.\n";
            shutdown();
            return false;
        }
        return true;
    }

    void shutdown() {
        if (stream) CUDA_CHECK(cudaStreamSynchronize(stream));

        if (graphExec) { cudaGraphExecDestroy(graphExec); graphExec = nullptr; }
        if (graph) { cudaGraphDestroy(graph);         graph = nullptr; }
        graphReady = false;

#if AI_HAVE_CUDA_KERNELS
        if (dGatherIdx) { cudaFree(dGatherIdx);    dGatherIdx = nullptr; }
        if (dGatherLogits) { cudaFree(dGatherLogits); dGatherLogits = nullptr; }
#endif

        if (dInput) { cudaFree(dInput);  dInput = nullptr; }
        if (dPolicy) { cudaFree(dPolicy); dPolicy = nullptr; }
        if (dValue) { cudaFree(dValue);  dValue = nullptr; }

        // Destroy aux streams
        for (cudaStream_t s : auxStreams) {
            if (s) cudaStreamDestroy(s);
        }
        auxStreams.clear();
        nbAuxStreams = 0;

        if (stream) { cudaStreamDestroy(stream); stream = nullptr; }

#if AI_HAVE_CUDA_KERNELS
        if (hGatherIdxPinned) { cudaFreeHost(hGatherIdxPinned);    hGatherIdxPinned = nullptr; }
        if (hGatherLogitsPinned) { cudaFreeHost(hGatherLogitsPinned); hGatherLogitsPinned = nullptr; }
#endif

        if (hInputPinned) { cudaFreeHost(hInputPinned);  hInputPinned = nullptr; }
        if (hPolicyPinned) { cudaFreeHost(hPolicyPinned); hPolicyPinned = nullptr; }
        if (hValuePinned) { cudaFreeHost(hValuePinned);  hValuePinned = nullptr; }

        if (ctx) { delete ctx;     ctx = nullptr; }
        if (engine) { delete engine;  engine = nullptr; }
        if (runtime) { delete runtime; runtime = nullptr; }
    }

    bool runBatchAndSync(int B) {
        if (!ctx || !stream || B <= 0 || B > maxBatch) return false;

        // Fast path: captured graph only for exact 256
        if (B == TRT_MAX_BATCH && graphReady && graphExec) {
            if (!ensureShape(TRT_MAX_BATCH)) return false;
            CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            return true;
        }

        if (!ensureShape(B)) return false;

        if (nbAuxStreams > 0 && (int)auxStreams.size() == nbAuxStreams) {
            ctx->setAuxStreams(auxStreams.data(), (int32_t)auxStreams.size());
        }

        CUDA_CHECK(cudaMemcpyAsync(dInput, hInputPinned, inBytes(B),
            cudaMemcpyHostToDevice, stream));

#if AI_HAVE_CUDA_KERNELS
        CUDA_CHECK(cudaMemcpyAsync(dGatherIdx, hGatherIdxPinned, gatherIdxBytes(B),
            cudaMemcpyHostToDevice, stream));
#endif

        if (!ctx->enqueueV3(stream)) return false;

#if AI_HAVE_CUDA_KERNELS
        {
            const int total = B * AI_MAX_MOVES;
            launchGatherPolicyKernel((const float*)dPolicy,
                (const int*)dGatherIdx,
                (float*)dGatherLogits,
                total,
                stream);
            CUDA_CHECK(cudaGetLastError());
        }

        CUDA_CHECK(cudaMemcpyAsync(hGatherLogitsPinned, dGatherLogits, gatherLogitsBytes(B),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(hValuePinned, dValue, valBytes(B),
            cudaMemcpyDeviceToHost, stream));
#else
        CUDA_CHECK(cudaMemcpyAsync(hPolicyPinned, dPolicy, polBytes(B),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(hValuePinned, dValue, valBytes(B),
            cudaMemcpyDeviceToHost, stream));
#endif

        CUDA_CHECK(cudaStreamSynchronize(stream));
        return true;
    }

    // Baseline infer: positions only (fills gather idx with zeros if enabled).
    bool inferBatch(const Position* posArr, int B) {
        if (!ctx || B <= 0 || B > maxBatch) return false;

        for (int i = 0; i < B; ++i) {
            auto* dst = reinterpret_cast<NNInput*>(
                hInputPinned + (size_t)i * (size_t)NN_INPUT_SIZE
                );
            positionToNNInput(posArr[i], *dst);
        }

#if AI_HAVE_CUDA_KERNELS
        std::fill_n(hGatherIdxPinned, (size_t)B * (size_t)AI_MAX_MOVES, -1);
#endif

        bool ok = runBatchAndSync(B);
        if (!ok) return false;

        for (int i = 0; i < B; ++i) hValuePinned[(size_t)i] = clamp01(hValuePinned[(size_t)i]);
        return true;
    }

    // Compatibility wrapper: get value + full policy CHW (slow-ish; for debug / main printing).
    bool inferBatch(const Position* posArr, int B,
        float* outValue,
        float* outPolicyChSq) {
        bool ok = inferBatch(posArr, B);
        if (!ok) return false;

        if (outPolicyChSq) {
            // Copy full policy D2H on-demand (first B only)
            CUDA_CHECK(cudaMemcpyAsync(hPolicyPinned, dPolicy,
                (size_t)B * (size_t)POLICY_SIZE * sizeof(float),
                cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            std::memcpy(outPolicyChSq, hPolicyPinned,
                (size_t)B * (size_t)POLICY_SIZE * sizeof(float));
        }

        if (outValue) {
            for (int i = 0; i < B; ++i) outValue[i] = hValuePinned[(size_t)i];
        }
        return true;
    }

    // Fast path used by MCTS server: fill input + per-move gather indices from PendingNN batch.
    bool inferBatchGather(const PendingNN* jobs, int B);
    bool inferBatchGather(const PendingNN* const* jobs, int B);
};

static TrtRunner g_trt;
static bool g_trtReady = false;
static int g_nnBatch = TRT_MAX_BATCH;

// Second TRT runner (own context/stream/buffers/graph). Used by the play-time
// inference server to pipeline CPU encode/expand phases with GPU compute:
// while one consumer thread waits on its GPU batch, the other runs its CPU phases.
static TrtRunner g_trt2;
static bool g_trt2Ready = false;

// ============================================================
// =============== Batched MultiThread MCTS ===================
// Leaf expansion uses a dedicated inference server thread.
// ============================================================

static AI_FORCEINLINE void cpuRelax() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#elif defined(__x86_64__) || defined(__i386)
    _mm_pause();
#else
    std::this_thread::yield();
#endif
}
// -------------------------------
// Time-based backoff wait helpers
// -------------------------------
static constexpr int64_t AI_LOCK_WAIT_US = 20000;
static constexpr int64_t AI_EXPAND_WAIT_US = 1000000;
static constexpr int64_t AI_SUBMIT_WAIT_US = 200;  // short timed wait for cancelable submit

static AI_FORCEINLINE void backoffWait(int& spins) {
    cpuRelax();
    ++spins;

    // IMPORTANT: no sleep_for(microseconds) — on many OSes this degrades to ~1ms.
    // Yield rarely so we do not lose throughput.
    if (spins == 256 || spins == 1024 || spins == 4096) {
        std::this_thread::yield();
    }
    if (spins > 16384) {
        // if it takes too long, start yielding more often, but still without sleeping
        std::this_thread::yield();
    }
}
// Queue throttling WITHOUT millisecond sleeps.
// spins is thread-local-ish counter to make backoffWait() behave nicely.
static AI_FORCEINLINE void throttleOnNNQueue_NoSleep(int qs, int& spins) {
    // thresholds tuned to your existing numbers
    if (qs <= 1100) { spins = 0; return; }

    // moderate overload: small backoff
    if (qs <= 1300) {
        for (int i = 0; i < 64; ++i) backoffWait(spins);
        return;
    }

    // heavy overload: stronger backoff (still sub-ms)
    if (qs <= 1600) {
        for (int i = 0; i < 256; ++i) backoffWait(spins);
        return;
    }

    // extreme overload: really back off (but no sleep_for(ms))
    for (int i = 0; i < 1024; ++i) backoffWait(spins);
}
template<class Clock = std::chrono::steady_clock>
static AI_FORCEINLINE bool notExpired(typename Clock::time_point deadline) {
    return Clock::now() < deadline;
}
static AI_FORCEINLINE bool waitWhileExpanding(const TTNode* n) {
    using Clock = std::chrono::steady_clock;
    const auto deadline = Clock::now() + std::chrono::microseconds(AI_EXPAND_WAIT_US);

    int spins = 0;
    while (n->expanded.load(std::memory_order_acquire) == 2) {
        if (!notExpired<Clock>(deadline)) return false;
        backoffWait(spins);
    }
    return true;
}

struct alignas(64) MCTSSlot {
    // meta: [gen:32][tag:32], tag: 0 empty, 1 locked, >=2 fingerprint
    std::atomic<uint64_t> meta{ 0 };
    uint32_t pad = 0;
    TTNode node;
};

struct MCTSTable {
    static constexpr uint32_t TAG_EMPTY32 = 0;
    static constexpr uint32_t TAG_LOCKED32 = 1;
    static constexpr int PROBE_LIMIT = 512;

    // Increment to "clear" table in O(1)
    std::atomic<uint32_t> curGen{ 1 };

    // Abort flag on overflow
    // oomCode: 0=ok, 1=node overflow (probe limit), 2=edge overflow
    std::atomic<bool>     abort{ false };
    std::atomic<uint32_t> oomCode{ 0 };

    std::vector<MCTSSlot> slots;
    uint64_t mask = 0;

    std::vector<TTEdge> edges;
    std::atomic<uint32_t> edgeTop{ 0 };

    explicit MCTSTable(size_t nodePow2, size_t edgeCap)
        : slots(nodePow2),
        mask((uint64_t)nodePow2 - 1),
        edges(edgeCap) {
    }

    // -------- meta helpers --------
    static AI_FORCEINLINE uint32_t metaGen(uint64_t meta) {
        return (uint32_t)(meta >> 32);
    }
    static AI_FORCEINLINE uint32_t metaTag(uint64_t meta) {
        return (uint32_t)(meta & 0xFFFFFFFFu);
    }
    static AI_FORCEINLINE uint64_t packMeta(uint32_t gen, uint32_t tag) {
        return (uint64_t(gen) << 32) | uint64_t(tag);
    }

    // 32-bit fingerprint (>=2)
    static AI_FORCEINLINE uint32_t makeTag32(uint64_t key) {
        uint32_t x = (uint32_t)key ^ (uint32_t)(key >> 32);

        // cheap avalanche mix
        x ^= x >> 16;
        x *= 0x7feb352dU;
        x ^= x >> 15;
        x *= 0x846ca68bU;
        x ^= x >> 16;

        // ensure >= 2 (so it can't be EMPTY/LOCKED)
        x |= 2u;
        if (x < 2u) x += 2u;
        return x;
    }

    // O(1) reset between games (tree reuse inside a game stays the same)
    void newGame() {
        // Reset edge allocator
        edgeTop.store(0, std::memory_order_relaxed);

        // Reset abort markers
        abort.store(false, std::memory_order_relaxed);
        oomCode.store(0, std::memory_order_relaxed);

        // Increment generation
        uint32_t g = curGen.fetch_add(1, std::memory_order_acq_rel) + 1u;

        // Extremely rare overflow: fall back to full clear once.
        if (g == 0u) {
            curGen.store(1u, std::memory_order_release);
            for (auto& s : slots) {
                s.meta.store(0u, std::memory_order_relaxed);
            }
        }
    }

    AI_FORCEINLINE TTEdge* edgePtr(uint32_t begin) { return &edges[(size_t)begin]; }

    AI_FORCEINLINE bool allocEdges(uint32_t count, uint32_t& outBegin) {
        if (AI_UNLIKELY(abort.load(std::memory_order_relaxed))) return false;

        uint32_t b = edgeTop.fetch_add(count, std::memory_order_relaxed);
        if ((size_t)b + (size_t)count > edges.size()) {
            abort.store(true, std::memory_order_release);
            oomCode.store(2u, std::memory_order_release);
            return false;
        }
        outBegin = b;
        return true;
    }

    // Insert-or-get
    AI_FORCEINLINE TTNode* getNode(uint64_t key) {
        if (AI_UNLIKELY(abort.load(std::memory_order_relaxed))) return nullptr;

        const uint32_t g = curGen.load(std::memory_order_acquire);
        const uint32_t wantTag = makeTag32(key);

        uint64_t idx = key & mask;
        int probe = 0;

        using Clock = std::chrono::steady_clock;
        const auto waitBudget = std::chrono::microseconds(AI_LOCK_WAIT_US);
        Clock::time_point lockStart{};
        uint64_t lockStartIdx = ~0ull;
        int lockSpins = 0;

        while (probe < PROBE_LIMIT) {
            MCTSSlot& s = slots[(size_t)idx];

            uint64_t m = s.meta.load(std::memory_order_acquire);
            const uint32_t mg = metaGen(m);
            const uint32_t mt = metaTag(m);

            // older generation => treat as empty, try to claim
            if (AI_UNLIKELY(mg != g)) {
                uint64_t expected = m;
                const uint64_t lockedMeta = packMeta(g, TAG_LOCKED32);

                if (s.meta.compare_exchange_weak(expected, lockedMeta,
                    std::memory_order_acq_rel,
                    std::memory_order_relaxed)) {
                    TTNode& n = s.node;

                    n.key = key;
                    n.edgeBegin = 0;
                    n.edgeCount = 0;
                    n.terminal = 0;
                    n.chance = 0;
                    n.expanded.store(0, std::memory_order_relaxed);
                    n.valueSum.store(0.0, std::memory_order_relaxed);
                    n.visits.store(0, std::memory_order_relaxed);
                    n.chanceCursor.store(0, std::memory_order_relaxed);

                    s.meta.store(packMeta(g, wantTag), std::memory_order_release);
                    return &n;
                }

                cpuRelax();
                continue; // retry same slot
            }

            if (mt == TAG_LOCKED32) {
                if (lockStartIdx != idx) {
                    lockStartIdx = idx;
                    lockStart = Clock::now();
                    lockSpins = 0;
                }

                if (Clock::now() - lockStart > waitBudget) {
                    return nullptr;
                }

                backoffWait(lockSpins);
                continue; // IMPORTANT: same slot, no probe++
            }

            lockStartIdx = ~0ull;
            lockSpins = 0;

            if (mt == wantTag) {
                if (AI_LIKELY(s.node.key == key)) return &s.node;
                // rare tag collision -> keep probing
            }

            if (mt == TAG_EMPTY32) {
                uint64_t expected = packMeta(g, TAG_EMPTY32);
                const uint64_t lockedMeta = packMeta(g, TAG_LOCKED32);

                if (s.meta.compare_exchange_weak(expected, lockedMeta,
                    std::memory_order_acq_rel,
                    std::memory_order_relaxed)) {
                    TTNode& n = s.node;

                    n.key = key;
                    n.edgeBegin = 0;
                    n.edgeCount = 0;
                    n.terminal = 0;
                    n.chance = 0;
                    n.expanded.store(0, std::memory_order_relaxed);
                    n.valueSum.store(0.0, std::memory_order_relaxed);
                    n.visits.store(0, std::memory_order_relaxed);
                    n.chanceCursor.store(0, std::memory_order_relaxed);

                    s.meta.store(packMeta(g, wantTag), std::memory_order_release);
                    return &n;
                }

                cpuRelax();
                continue; // retry same slot
            }

            idx = (idx + 1) & mask;
            ++probe;
        }

        abort.store(true, std::memory_order_release);
        oomCode.store(1u, std::memory_order_release);
        return nullptr;
    }

    AI_FORCEINLINE TTNode* findNodeNoInsert(uint64_t key) {
        const uint32_t g = curGen.load(std::memory_order_acquire);
        const uint32_t wantTag = makeTag32(key);

        uint64_t idx = key & mask;
        int probe = 0;

        using Clock = std::chrono::steady_clock;
        const auto waitBudget = std::chrono::microseconds(AI_LOCK_WAIT_US);
        Clock::time_point lockStart{};
        uint64_t lockStartIdx = ~0ull;
        int lockSpins = 0;

        while (probe < PROBE_LIMIT) {
            MCTSSlot& s = slots[(size_t)idx];

            const uint64_t m = s.meta.load(std::memory_order_acquire);
            if (AI_UNLIKELY(metaGen(m) != g)) return nullptr;

            const uint32_t mt = metaTag(m);

            if (mt == TAG_LOCKED32) {
                if (lockStartIdx != idx) {
                    lockStartIdx = idx;
                    lockStart = Clock::now();
                    lockSpins = 0;
                }

                if (Clock::now() - lockStart > waitBudget) {
                    return nullptr;
                }

                backoffWait(lockSpins);
                continue; // IMPORTANT: same slot, no probe++
            }

            lockStartIdx = ~0ull;
            lockSpins = 0;

            if (mt == wantTag) {
                if (AI_LIKELY(s.node.key == key)) return &s.node;
                // rare tag collision -> probe further
            }

            if (mt == TAG_EMPTY32) return nullptr;

            idx = (idx + 1) & mask;
            ++probe;
        }

        return nullptr;
    }
};


static AI_FORCEINLINE const char* oomWhat(uint32_t oomCode) {
    switch (oomCode) {
    case 1u: return "node";
    case 2u: return "edge";
    default: return "unknown";
    }
}

static AI_FORCEINLINE float nodeQ(const TTNode& n) {
    uint32_t v = n.visits.load(std::memory_order_acquire);
    if (!v) return 0.5f;
    double s = n.valueSum.load(std::memory_order_relaxed);
    return clamp01((float)(s / (double)v));
}

static AI_FORCEINLINE float edgeQ(const TTEdge& e) {
    uint32_t v = e.visits.load(std::memory_order_acquire);
    if (!v) return -1.0f;
    double s = e.valueSum.load(std::memory_order_relaxed);
    return clamp01((float)(s / (double)v));
}

// PV selection: first max visits, then max Q, then max prior.
static AI_FORCEINLINE int selectBestPVEdge(const TTNode& n, const TTEdge* e0) {
    int bestI = 0;
    uint32_t bestV = 0;
    float bestQ = -2.0f;
    float bestP = -1.0f;

    for (int i = 0; i < (int)n.edgeCount; ++i) {
        const TTEdge& e = e0[i];
        uint32_t v = e.visits.load(std::memory_order_relaxed);
        float q = (v ? clamp01(e.sum() / (float)v) : -1.0f);
        float p = e.prior();

        if (v > bestV ||
            (v == bestV && (q > bestQ ||
                (q == bestQ && p > bestP)))) {
            bestI = i;
            bestV = v;
            bestQ = q;
            bestP = p;
        }
    }
    return bestI;
}

struct SearchParams {
    float c_init = 1.25f;
    float fpu_reduction = 0.08f;
    float c_base = 1000000;
    float c_mult = 1;
};

static const SearchParams kDefaultSearchParams{};

static SearchParams makeSearchParams(float c_init, float fpu_reduction) {
    SearchParams params = kDefaultSearchParams;
    params.c_init = c_init;
    params.fpu_reduction = fpu_reduction;
    return params;
}

static AI_FORCEINLINE float cpuctFromVisits(
    uint32_t parentVisits,
    bool isRoot,
    const SearchParams& sp) {
    float c = sp.c_init + sp.c_mult * log((parentVisits + sp.c_base) / sp.c_base);
    if (isRoot) c *= 1.10f;
    return c;
}

static AI_FORCEINLINE int selectPUCT(const TTNode& n,
    const TTEdge* e0,
    float cpuct,
    uint32_t parentVisits,
    float parentQ,
    const SearchParams& sp,
    uint32_t rngJitter) {

    const float sqrtN = std::sqrt((float)(parentVisits + 1u));

    float best = -1e30f;
    int bestI = 0;

    const int cnt = (int)n.edgeCount;
    for (int i = 0; i < cnt; ++i) {
        const TTEdge& e = e0[i];
        uint32_t ev = e.visits.load(std::memory_order_relaxed);
        const float p = e.prior();

        const float fpu = clamp01(parentQ - sp.fpu_reduction);
        const float q = ev ? clamp01((float)(e.sum() / (double)ev)) : fpu;

        const float u = cpuct * p * (sqrtN / (1.0f + (float)ev));

        const float jit = (float)((rngJitter + (uint32_t)i * 2654435761u) & 1023u)
            * (1.0f / 1023.0f) * 1e-6f;

        const float s = q + u + jit;
        if (s > best) {
            best = s;
            bestI = i;
        }
    }
    (void)n;
    return bestI;
}

static constexpr int MCTS_MAX_DEPTH = 256;

// Classic virtual loss
static constexpr uint32_t VLOSS_N = 1;     // usually 1; 2-3 only makes sense with a very large number of threads
static constexpr float    VLOSS_VALUE = 0.0f; // value on the [0..1] scale; 0.0 = "loss for side-to-move"
static constexpr bool     VLOSS_BUMP_NODE_VISITS = false; // optional

struct TraceStep {
    TTNode* node = nullptr;
    TTEdge* edge = nullptr;
    bool flip = false;

    bool vloss = false;
};

struct Trace {
    int n = 0;
    TraceStep st[MCTS_MAX_DEPTH];

    AI_FORCEINLINE Trace() = default;

    // copy only used prefix [0..n)
    AI_FORCEINLINE Trace(const Trace& o) : n(o.n) {
        if (n > 0) {
            std::memcpy(st, o.st, (size_t)n * sizeof(TraceStep));
        }
    }

    AI_FORCEINLINE Trace& operator=(const Trace& o) {
        if (this != &o) {
            n = o.n;
            if (n > 0) {
                std::memcpy(st, o.st, (size_t)n * sizeof(TraceStep));
            }
        }
        return *this;
    }

    // move = same cheap prefix copy, no heap
    AI_FORCEINLINE Trace(Trace&& o) noexcept : n(o.n) {
        if (n > 0) {
            std::memcpy(st, o.st, (size_t)n * sizeof(TraceStep));
        }
        o.n = 0;
    }

    AI_FORCEINLINE Trace& operator=(Trace&& o) noexcept {
        if (this != &o) {
            n = o.n;
            if (n > 0) {
                std::memcpy(st, o.st, (size_t)n * sizeof(TraceStep));
            }
            o.n = 0;
        }
        return *this;
    }

    AI_FORCEINLINE void reset() { n = 0; }

    AI_FORCEINLINE void copyFrom(const Trace& o) {
        n = o.n;
        if (n > 0) {
            std::memcpy(st, o.st, (size_t)n * sizeof(TraceStep));
        }
    }

    AI_FORCEINLINE TraceStep& push(TTNode* node, TTEdge* edge, bool flip, bool vloss) {
        assert(n >= 0 && n < MCTS_MAX_DEPTH);

        if (AI_UNLIKELY((unsigned)n >= (unsigned)MCTS_MAX_DEPTH)) {
            std::cerr << "[FATAL] Trace overflow: n=" << n
                << " MCTS_MAX_DEPTH=" << MCTS_MAX_DEPTH << "\n";
            std::abort();
        }

        TraceStep& s = st[n];
        s.node = node;
        s.edge = edge;
        s.flip = flip;
        s.vloss = vloss;
        ++n;
        return s;
    }
};

struct SearchWaitGroup {
    std::atomic<int> pending{ 0 };
    std::mutex m;
    std::condition_variable cv;
};

static AI_FORCEINLINE void waitGroupAdd(SearchWaitGroup* wg) {
    if (!wg) return;
    wg->pending.fetch_add(1, std::memory_order_relaxed);
}

static AI_FORCEINLINE void waitGroupDone(SearchWaitGroup* wg) {
    if (!wg) return;
    int prev = wg->pending.fetch_sub(1, std::memory_order_acq_rel);
    if (prev == 1) {
        std::lock_guard<std::mutex> lk(wg->m);
        wg->cv.notify_all();
    }
}

// Bounded wait: returns true if pending reached zero within the timeout.
static bool waitGroupWaitZeroFor(SearchWaitGroup* wg, std::chrono::milliseconds timeout) {
    if (!wg) return true;
    std::unique_lock<std::mutex> lk(wg->m);
    return wg->cv.wait_for(lk, timeout, [&] {
        return wg->pending.load(std::memory_order_acquire) == 0;
        });
}

static void waitGroupWaitZero(SearchWaitGroup* wg) {
    if (!wg) return;
    std::unique_lock<std::mutex> lk(wg->m);
    int stuckMinutes = 0;
    while (!wg->cv.wait_for(lk, std::chrono::seconds(60), [&] {
        return wg->pending.load(std::memory_order_acquire) == 0;
        })) {
        ++stuckMinutes;
        std::cerr << "[waitGroup] STUCK for " << stuckMinutes
            << " min: pending=" << wg->pending.load(std::memory_order_relaxed)
            << " (likely a lost NN job; search cannot finish)" << std::endl;
    }
}

// ---- global progress heartbeat + process watchdog ----
// Any real forward progress (a finished self-play game, an arena progress line,
// a training loop iteration) refreshes this. The watchdog kills the process if
// it goes stale: full state is persisted hourly, so a restart is cheap, while a
// silent deadlock costs days.
static std::atomic<uint64_t> g_progressHeartbeatMs{ 0 };

static AI_FORCEINLINE uint64_t steadyNowMs() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static AI_FORCEINLINE void noteTrainingProgress() {
    g_progressHeartbeatMs.store(steadyNowMs(), std::memory_order_relaxed);
}

struct PendingNN {
    MCTSTable* ownerT = nullptr;          // owner table for shared inference server
    SearchWaitGroup* waitGroup = nullptr; // per-search completion tracking

    TTNode* leaf = nullptr;
    Position pos;
    MoveList ml;
    Trace trace;

    // precomputed CHW policy indices for ml.m[0..ml.n)
    std::array<uint16_t, AI_MAX_MOVES> policyIdx{};
};

static constexpr uint16_t INVALID_POLICY_IDX = 0xFFFFu;

// ============================================================
// PendingNN object pool
// ============================================================

static std::mutex g_pendingMutex;
static std::vector<std::unique_ptr<PendingNN>> g_pendingPool;

// global pool cap
static constexpr size_t AI_MAX_PENDING_POOL = 4096;

// block allocator params
static constexpr size_t AI_PENDING_BLOCK_SIZE = 64;
static constexpr size_t AI_PENDING_TLS_KEEP = 128;

// per-thread local cache
static thread_local std::vector<std::unique_ptr<PendingNN>> g_pendingTLS;

static AI_FORCEINLINE void resetPendingNN(PendingNN& p) {
    p.ownerT = nullptr;
    p.waitGroup = nullptr;
    p.leaf = nullptr;
    p.ml.n = 0;
    p.trace.reset();
    p.policyIdx.fill(INVALID_POLICY_IDX);
}

static AI_FORCEINLINE void completePendingNNJob(PendingNN& p) {
    if (p.waitGroup) {
        waitGroupDone(p.waitGroup);
        p.waitGroup = nullptr;
    }
    p.ownerT = nullptr;
}

static AI_FORCEINLINE void refillPendingTLSIfNeeded() {
    if (!g_pendingTLS.empty()) return;

    // First try to grab a block from the global pool.
    {
        std::lock_guard<std::mutex> lk(g_pendingMutex);

        const size_t take = std::min(AI_PENDING_BLOCK_SIZE, g_pendingPool.size());
        g_pendingTLS.reserve(AI_PENDING_TLS_KEEP);

        for (size_t i = 0; i < take; ++i) {
            g_pendingTLS.push_back(std::move(g_pendingPool.back()));
            g_pendingPool.pop_back();
        }
    }

    if (!g_pendingTLS.empty()) return;

    // Global pool empty: allocate a fresh local block.
    g_pendingTLS.reserve(AI_PENDING_TLS_KEEP);
    for (size_t i = 0; i < AI_PENDING_BLOCK_SIZE; ++i) {
        auto p = std::make_unique<PendingNN>();
        resetPendingNN(*p);
        g_pendingTLS.push_back(std::move(p));
    }
}

static AI_FORCEINLINE void flushPendingTLSPartial() {
    // Keep one block locally, flush excess to global pool.
    if (g_pendingTLS.size() <= AI_PENDING_TLS_KEEP) return;

    std::lock_guard<std::mutex> lk(g_pendingMutex);

    while (g_pendingTLS.size() > AI_PENDING_BLOCK_SIZE &&
        g_pendingPool.size() < AI_MAX_PENDING_POOL) {
        g_pendingPool.push_back(std::move(g_pendingTLS.back()));
        g_pendingTLS.pop_back();
    }
}

static std::unique_ptr<PendingNN> allocPendingNN() {
    refillPendingTLSIfNeeded();

    auto p = std::move(g_pendingTLS.back());
    g_pendingTLS.pop_back();

    resetPendingNN(*p);
    return p;
}

static void freePendingNN(std::unique_ptr<PendingNN> p) {
    if (!p) return;

    completePendingNNJob(*p);
    resetPendingNN(*p);
    g_pendingTLS.push_back(std::move(p));
    flushPendingTLSPartial();
}

template<class TVec>
static void freePendingBatch(TVec& jobs) {
    for (auto& p : jobs) {
        if (!p) continue;
        completePendingNNJob(*p);
        resetPendingNN(*p);

        g_pendingTLS.push_back(std::move(p));

        // Flush in chunks if batch is large.
        if (g_pendingTLS.size() > (AI_PENDING_TLS_KEEP + AI_PENDING_BLOCK_SIZE)) {
            flushPendingTLSPartial();
        }
    }

    jobs.clear();
    flushPendingTLSPartial();
}

static AI_FORCEINLINE float pendingPolicyLogitFromFullCHW(
    const PendingNN& p,
    int i,
    const float* polChSq) {
    const uint16_t k = p.policyIdx[(size_t)i];

    if (AI_UNLIKELY(k == INVALID_POLICY_IDX || (unsigned)k >= (unsigned)POLICY_SIZE)) {
        // This should be impossible, but if mapping breaks —
        // assign almost -inf so softmax gives ~0.
        return -1e30f;
    }

    return polChSq[(size_t)k];
}
static AI_FORCEINLINE void fillPendingPolicyIdx(PendingNN& p) {
    const int n = p.ml.n;

    for (int i = 0; i < n; ++i) {
        const int k = policyIndexCHWCanonical(p.ml.m[i], p.pos);
        p.policyIdx[(size_t)i] =
            ((unsigned)k < (unsigned)POLICY_SIZE) ? (uint16_t)k : INVALID_POLICY_IDX;
    }

    for (int i = n; i < AI_MAX_MOVES; ++i) {
        p.policyIdx[(size_t)i] = INVALID_POLICY_IDX;
    }
}

static AI_FORCEINLINE bool tryAddVisitsNoOverflow(std::atomic<uint32_t>& visits, uint32_t delta) {
    uint32_t old = visits.load(std::memory_order_relaxed);
    for (;;) {
        if (AI_UNLIKELY(old > (std::numeric_limits<uint32_t>::max() - delta))) {
            return false;
        }
        if (visits.compare_exchange_weak(old, old + delta,
            std::memory_order_release,
            std::memory_order_relaxed)) {
            return true;
        }
    }
}

static AI_FORCEINLINE bool tryAddVisitAndValueNoOverflow(TTNode* node, float v) {
    if (!node) return true;
    if (!tryAddVisitsNoOverflow(node->visits, 1)) return false;
    atomicAddDouble(node->valueSum, (double)v);
    return true;
}

static AI_FORCEINLINE bool tryAddVisitAndValueNoOverflow(TTEdge* edge, float v) {
    if (!edge) return true;
    if (!tryAddVisitsNoOverflow(edge->visits, 1)) return false;
    atomicAddDouble(edge->valueSum, (double)v);
    return true;
}

static AI_FORCEINLINE bool applyVirtualLoss(TraceStep& s) {
    if (!s.vloss) return true;

    if (VLOSS_BUMP_NODE_VISITS && s.node) {
        const uint32_t nv = s.node->visits.load(std::memory_order_relaxed);
        if (AI_UNLIKELY(nv > (std::numeric_limits<uint32_t>::max() - VLOSS_N))) return false;
    }

    if (s.edge) {
        const uint32_t ev = s.edge->visits.load(std::memory_order_relaxed);
        if (AI_UNLIKELY(ev > (std::numeric_limits<uint32_t>::max() - VLOSS_N))) return false;
    }

    if (VLOSS_BUMP_NODE_VISITS && s.node) {
        s.node->visits.fetch_add(VLOSS_N, std::memory_order_relaxed);
        // do NOT touch node valueSum (classic approach)
    }

    if (s.edge) {
        s.edge->visits.fetch_add(VLOSS_N, std::memory_order_relaxed);
        // “loss” on [0..1] scale => add W as if VLOSS_VALUE returned
        if (VLOSS_VALUE != 0.0f) {
            atomicAddDouble(s.edge->valueSum, (double)VLOSS_VALUE * (double)VLOSS_N);
        }
        // if VLOSS_VALUE=0.0f, valueSum can be left untouched
    }

    return true;
}

static AI_FORCEINLINE void rollbackVirtualLoss(Trace& tr) {
    for (int i = 0; i < tr.n; ++i) {
        TraceStep& s = tr.st[i];
        if (!s.vloss) continue;

        // edge rollback
        if (s.edge) {
            if (VLOSS_VALUE != 0.0f) {
                atomicAddDouble(s.edge->valueSum, -((double)VLOSS_VALUE * (double)VLOSS_N));
            }
            s.edge->visits.fetch_sub(VLOSS_N, std::memory_order_relaxed);
        }

        // node rollback
        if (VLOSS_BUMP_NODE_VISITS && s.node) {
            s.node->visits.fetch_sub(VLOSS_N, std::memory_order_relaxed);
        }

        s.vloss = false;
    }
}

static AI_FORCEINLINE void cancelPendingNN(PendingNN& p) {
    // undo virtual loss first
    rollbackVirtualLoss(p.trace);

    // release leaf if we claimed expansion but never finished it
    if (p.leaf) {
        uint8_t ex = p.leaf->expanded.load(std::memory_order_acquire);
        if (ex == 2) {
            p.leaf->expanded.store(0, std::memory_order_release);
        }
    }

    // IMPORTANT:
    // do NOT clear ownerT / waitGroup here.
    // Caller may still need completePendingNNJob(p).
    p.leaf = nullptr;
    p.ml.n = 0;
    p.trace.reset();
    p.policyIdx.fill(INVALID_POLICY_IDX);
}

struct ExpansionClaimGuard {
    TTNode* node = nullptr;
    bool active = false;

    ExpansionClaimGuard() = default;
    explicit ExpansionClaimGuard(TTNode* n) noexcept
        : node(n), active(n != nullptr) {
    }

    void arm(TTNode* n) noexcept {
        node = n;
        active = (n != nullptr);
    }

    void release() noexcept {
        active = false;
    }

    ~ExpansionClaimGuard() noexcept {
        if (!active || !node) return;

        if (node->expanded.load(std::memory_order_acquire) == 2) {
            node->expanded.store(0, std::memory_order_release);
        }
    }

    ExpansionClaimGuard(const ExpansionClaimGuard&) = delete;
    ExpansionClaimGuard& operator=(const ExpansionClaimGuard&) = delete;
};

struct PendingNNGuard {
    PendingNN* p = nullptr;
    bool active = false;

    PendingNNGuard() = default;
    explicit PendingNNGuard(PendingNN& ref) noexcept
        : p(&ref), active(true) {
    }

    void reset(PendingNN& ref) noexcept {
        p = &ref;
        active = true;
    }

    void release() noexcept {
        active = false;
    }

    ~PendingNNGuard() noexcept {
        if (!active || !p) return;
        cancelPendingNN(*p);
        completePendingNNJob(*p);
    }

    PendingNNGuard(const PendingNNGuard&) = delete;
    PendingNNGuard& operator=(const PendingNNGuard&) = delete;
};

struct PendingNNPtrGuard {
    std::unique_ptr<PendingNN>* up = nullptr;
    bool active = false;

    PendingNNPtrGuard() = default;
    explicit PendingNNPtrGuard(std::unique_ptr<PendingNN>& p) noexcept
        : up(&p), active(true) {
    }

    void reset(std::unique_ptr<PendingNN>& p) noexcept {
        up = &p;
        active = true;
    }

    void release() noexcept {
        active = false;
    }

    ~PendingNNPtrGuard() noexcept {
        if (!active || !up || !(*up)) return;
        cancelPendingNN(**up);
        completePendingNNJob(**up);
        freePendingNN(std::move(*up));
    }

    PendingNNPtrGuard(const PendingNNPtrGuard&) = delete;
    PendingNNPtrGuard& operator=(const PendingNNPtrGuard&) = delete;
};

static AI_FORCEINLINE void abortPendingNNInferFailure(
    PendingNN& p,
    MCTSTable* fallbackOwner,
    const char* where)
{
    (void)where;
    MCTSTable* owner = p.ownerT ? p.ownerT : fallbackOwner;
    if (owner) {
        owner->abort.store(true, std::memory_order_release);
    }

    cancelPendingNN(p);
    completePendingNNJob(p);
}

static AI_FORCEINLINE void backprop(TTNode* leaf, float v, Trace& tr, MCTSTable* ownerT = nullptr) {
    rollbackVirtualLoss(tr);

    if (AI_UNLIKELY(!tryAddVisitAndValueNoOverflow(leaf, v))) {
        if (ownerT) ownerT->abort.store(true, std::memory_order_release);
    }

    for (int i = tr.n - 1; i >= 0; --i) {
        TraceStep& s = tr.st[i];
        if (s.flip) v = 1.0f - v;
        if (s.edge && AI_UNLIKELY(!tryAddVisitAndValueNoOverflow(s.edge, v))) {
            if (ownerT) ownerT->abort.store(true, std::memory_order_release);
        }
        if (AI_UNLIKELY(!tryAddVisitAndValueNoOverflow(s.node, v))) {
            if (ownerT) ownerT->abort.store(true, std::memory_order_release);
        }
    }
}

static constexpr float ROOT_DIR_EPS = 0.25f;
static constexpr float ROOT_DIR_ALPHA = 0.30f;

static AI_FORCEINLINE void renormProbs(std::vector<float>& p) {
    double s = 0.0;
    for (float v : p) s += (double)v;
    if (!(s > 0.0)) return;
    float inv = (float)(1.0 / s);
    for (float& v : p) v *= inv;
}

static void applyRootDirichletNoise(std::vector<float>& priors) {
    if (priors.size() < 2) return;

    std::gamma_distribution<float> gamma(ROOT_DIR_ALPHA, 1.0f);
    float sum = 0.0f;

    std::vector<float> noise(priors.size());
    for (size_t i = 0; i < priors.size(); ++i) {
        float x = gamma(Random);
        if (!(x > 0.0f)) x = 0.0f;
        noise[i] = x;
        sum += x;
    }
    if (!(sum > 0.0f)) return;

    float inv = 1.0f / sum;
    for (size_t i = 0; i < priors.size(); ++i) {
        float n = noise[i] * inv;
        priors[i] = (1.0f - ROOT_DIR_EPS) * priors[i] + ROOT_DIR_EPS * n;
    }
    renormProbs(priors);
}

static AI_FORCEINLINE void publishReady(TTNode* n,
    uint64_t key,
    uint32_t begin,
    uint8_t count,
    uint8_t terminal,
    uint8_t chance) {
    n->key = key;
    n->edgeBegin = begin;
    n->edgeCount = count;
    n->terminal = terminal;
    n->chance = chance;
    n->expanded.store(1, std::memory_order_release);
}

static AI_FORCEINLINE void publishTerminalWithMove(MCTSTable& T,
    TTNode* n,
    uint64_t key,
    int move) {
    uint32_t begin = 0;
    if (move != 0 && T.allocEdges(1, begin)) {
        TTEdge& e = T.edges[(size_t)begin];
        e.move = (uint16_t)move;
        e.setPrior(1.0f);
        e.valueSum.store(0.0f, std::memory_order_relaxed);
        e.visits.store(0, std::memory_order_relaxed);
        publishReady(n, key, begin, 1, 1, 0);
        return;
    }
    publishReady(n, key, 0, 0, 1, 0);
}

// Old expansion: policy logits in CHW [pl][sq]
// Fixed expansion: clamp priors BEFORE renorm, and write e.prior ONCE (no later overwrite).
// ===== stack-only helpers (NO heap) =====

static AI_FORCEINLINE void renormProbsArr(float* p, int n) {
    if (n <= 0) return;
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += (double)p[i];
    if (!(s > 0.0)) return;
    float inv = (float)(1.0 / s);
    for (int i = 0; i < n; ++i) p[i] *= inv;
}

static AI_FORCEINLINE void softmaxLocalArr(float* x, int n) {
    if (n <= 0) return;

    float mx = x[0];
    for (int i = 1; i < n; ++i) if (x[i] > mx) mx = x[i];

    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        float v = std::exp(x[i] - mx);
        x[i] = v;
        sum += (double)v;
    }
    if (!(sum > 0.0)) {
        float inv = 1.0f / (float)n;
        for (int i = 0; i < n; ++i) x[i] = inv;
        return;
    }
    float inv = (float)(1.0 / sum);
    for (int i = 0; i < n; ++i) x[i] *= inv;
}

// Dirichlet noise (NO heap)
static void applyRootDirichletNoiseArr(float* priors, int n) {
    if (n < 2) return;

    std::gamma_distribution<float> gamma(ROOT_DIR_ALPHA, 1.0f);

    std::array<float, AI_MAX_MOVES> noise{};
    float sum = 0.0f;

    for (int i = 0; i < n; ++i) {
        float x = gamma(Random);
        if (!(x > 0.0f) || !std::isfinite(x)) x = 0.0f;
        noise[(size_t)i] = x;
        sum += x;
    }
    if (!(sum > 0.0f)) return;

    float inv = 1.0f / sum;
    for (int i = 0; i < n; ++i) {
        float d = noise[(size_t)i] * inv;
        priors[i] = (1.0f - ROOT_DIR_EPS) * priors[i] + ROOT_DIR_EPS * d;
    }

    renormProbsArr(priors, n);
}

struct RootNoiseBackup {
    TTEdge* e0 = nullptr;
    int n = 0;
    std::array<uint16_t, AI_MAX_MOVES> priorQ{};
};

static void applyTemporaryRootNoise(MCTSTable& T,
    const Position& rootPos,
    bool enabled,
    RootNoiseBackup& bk) {
    bk.e0 = nullptr;
    bk.n = 0;

    if (!enabled) return;

    TTNode* root = T.findNodeNoInsert(rootPos.key);
    if (!root) return;
    if (root->expanded.load(std::memory_order_acquire) != 1) return;
    if (root->edgeCount < 2) return;

    bk.e0 = T.edgePtr(root->edgeBegin);
    bk.n = (int)root->edgeCount;

    std::array<float, AI_MAX_MOVES> noisy{};
    for (int i = 0; i < bk.n; ++i) {
        bk.priorQ[(size_t)i] = bk.e0[i].priorRaw();
        noisy[(size_t)i] = bk.e0[i].prior();
    }

    applyRootDirichletNoiseArr(noisy.data(), bk.n);

    for (int i = 0; i < bk.n; ++i) {
        bk.e0[i].setPrior(noisy[(size_t)i]);
    }
}

static void restoreTemporaryRootNoise(RootNoiseBackup& bk) {
    if (!bk.e0 || bk.n <= 0) return;

    for (int i = 0; i < bk.n; ++i) {
        bk.e0[i].setPriorRaw(bk.priorQ[(size_t)i]);
    }

    bk.e0 = nullptr;
    bk.n = 0;
}

struct RootNoiseGuard {
    RootNoiseBackup bk;

    RootNoiseGuard(MCTSTable& T,
        const Position& rootPos,
        bool enabled) {
        applyTemporaryRootNoise(T, rootPos, enabled, bk);
    }

    ~RootNoiseGuard() noexcept {
        restoreTemporaryRootNoise(bk);
    }

    RootNoiseGuard(const RootNoiseGuard&) = delete;
    RootNoiseGuard& operator=(const RootNoiseGuard&) = delete;
};

struct AtomicStopGuard {
    std::atomic<bool>* flag = nullptr;

    explicit AtomicStopGuard(std::atomic<bool>& f) : flag(&f) {}

    ~AtomicStopGuard() noexcept {
        if (flag) flag->store(true, std::memory_order_relaxed);
    }

    void release() noexcept { flag = nullptr; }

    AtomicStopGuard(const AtomicStopGuard&) = delete;
    AtomicStopGuard& operator=(const AtomicStopGuard&) = delete;
};

struct ThreadJoinGuard {
    std::vector<std::thread>* threads = nullptr;

    explicit ThreadJoinGuard(std::vector<std::thread>& v) : threads(&v) {}

    ~ThreadJoinGuard() noexcept {
        if (!threads) return;
        for (auto& th : *threads) {
            if (th.joinable()) th.join();
        }
    }

    void release() noexcept { threads = nullptr; }

    ThreadJoinGuard(const ThreadJoinGuard&) = delete;
    ThreadJoinGuard& operator=(const ThreadJoinGuard&) = delete;
};

struct InferenceServer;

struct InferenceServerStopGuard {
    InferenceServer* srv = nullptr;

    explicit InferenceServerStopGuard(InferenceServer& s) : srv(&s) {}

    ~InferenceServerStopGuard() noexcept;

    void release() noexcept { srv = nullptr; }

    InferenceServerStopGuard(const InferenceServerStopGuard&) = delete;
    InferenceServerStopGuard& operator=(const InferenceServerStopGuard&) = delete;
};


// ============================================================
// Old expansion: policy logits in CHW [pl][sq] (4672 floats)
// New: priors stored in std::array<float,255>, no heap.
// ============================================================
static void expandLeafWithOutputs(MCTSTable& T,
    PendingNN& p,
    float v,
    const float* polChSq) {
    const uint32_t cntU = (uint32_t)p.ml.n;
    const int cnt = (int)cntU;

    // Safety (shouldn't happen, but prevents UB)
    if (cnt <= 0) {
        // no edges => treat as chance or dead end (here: dead end)
        p.leaf->key = p.pos.key;
        p.leaf->edgeBegin = 0;
        p.leaf->edgeCount = 0;
        p.leaf->terminal = 0;
        p.leaf->chance = 0;

        backprop(p.leaf, v, p.trace, &T);
        publishReady(p.leaf, p.pos.key, 0, 0, 0, 0);
        return;
    }

    // Build priors from full policy logits (CHW) using precomputed indices
    std::array<float, AI_MAX_MOVES> priors{};
    for (int i = 0; i < cnt; ++i) {
        priors[(size_t)i] = pendingPolicyLogitFromFullCHW(p, i, polChSq);
    }

    // Softmax over legal moves
    softmaxLocalArr(priors.data(), cnt);

    // Clamp priors, renorm, store ONCE
    for (int i = 0; i < cnt; ++i) {
        float& pr = priors[(size_t)i];
        if (!(pr > 0.0f) || !std::isfinite(pr)) pr = 1e-8f;
    }
    renormProbsArr(priors.data(), cnt);

    // Allocate edges
    uint32_t begin = 0;
    if (!T.allocEdges(cntU, begin)) {
        // edge overflow -> abort search, but still release node
        T.abort.store(true, std::memory_order_release);
        if (T.oomCode.load(std::memory_order_relaxed) == 0)
            T.oomCode.store(2u, std::memory_order_relaxed);

        p.leaf->key = p.pos.key;
        p.leaf->edgeBegin = 0;
        p.leaf->edgeCount = 0;
        p.leaf->terminal = 0;
        p.leaf->chance = 0;

        backprop(p.leaf, v, p.trace, &T);
        publishReady(p.leaf, p.pos.key, 0, 0, 0, 0);
        return;
    }

    // Init edges
    for (uint32_t i = 0; i < cntU; ++i) {
        TTEdge& e = T.edges[(size_t)begin + (size_t)i];
        e.move = p.ml.m[i];
        e.setPrior(priors[i]);
        e.valueSum.store(0.0f, std::memory_order_relaxed);
        e.visits.store(0, std::memory_order_relaxed);
    }

    // Publish node + backprop
    p.leaf->key = p.pos.key;
    p.leaf->edgeBegin = begin;
    p.leaf->edgeCount = (uint8_t)cntU;
    p.leaf->terminal = 0;
    p.leaf->chance = 0;

    backprop(p.leaf, v, p.trace, &T);
    publishReady(p.leaf, p.pos.key, begin, (uint8_t)cntU, 0, 0);
}

// ============================================================
// Gathered-logits expansion: logits already in move order [0..ml.n)
// New: priors stored in std::array<float,255>, no heap.
// ============================================================
static void expandLeafWithGatheredLogits(MCTSTable& T,
    PendingNN& p,
    float v,
    const float* logitsMoveOrder) {
    const uint32_t cntU = (uint32_t)p.ml.n;
    const int cnt = (int)cntU;

    if (cnt <= 0) {
        p.leaf->key = p.pos.key;
        p.leaf->edgeBegin = 0;
        p.leaf->edgeCount = 0;
        p.leaf->terminal = 0;
        p.leaf->chance = 0;

        backprop(p.leaf, v, p.trace, &T);
        publishReady(p.leaf, p.pos.key, 0, 0, 0, 0);
        return;
    }

    // Copy gathered logits into priors array
    std::array<float, AI_MAX_MOVES> priors{};
    for (int i = 0; i < cnt; ++i) {
        priors[(size_t)i] = logitsMoveOrder[i];
    }

    // Softmax over legal moves
    softmaxLocalArr(priors.data(), cnt);


    // Clamp priors, renorm, store ONCE
    for (int i = 0; i < cnt; ++i) {
        float& pr = priors[(size_t)i];
        if (!(pr > 0.0f) || !std::isfinite(pr)) pr = 1e-8f;
    }
    renormProbsArr(priors.data(), cnt);

    // Allocate edges
    uint32_t begin = 0;
    if (!T.allocEdges(cntU, begin)) {
        T.abort.store(true, std::memory_order_release);
        if (T.oomCode.load(std::memory_order_relaxed) == 0)
            T.oomCode.store(2u, std::memory_order_relaxed);

        p.leaf->key = p.pos.key;
        p.leaf->edgeBegin = 0;
        p.leaf->edgeCount = 0;
        p.leaf->terminal = 0;
        p.leaf->chance = 0;

        backprop(p.leaf, v, p.trace, &T);
        publishReady(p.leaf, p.pos.key, 0, 0, 0, 0);
        return;
    }

    // Init edges
    for (uint32_t i = 0; i < cntU; ++i) {
        TTEdge& e = T.edges[(size_t)begin + (size_t)i];
        e.move = p.ml.m[i];
        e.setPrior(priors[i]);
        e.valueSum.store(0.0f, std::memory_order_relaxed);
        e.visits.store(0, std::memory_order_relaxed);
    }

    // Publish node + backprop
    p.leaf->key = p.pos.key;
    p.leaf->edgeBegin = begin;
    p.leaf->edgeCount = (uint8_t)cntU;
    p.leaf->terminal = 0;
    p.leaf->chance = 0;

    backprop(p.leaf, v, p.trace, &T);
    publishReady(p.leaf, p.pos.key, begin, (uint8_t)cntU, 0, 0);
}

// ============================================================
// TrtRunner::inferBatchGather (needs PendingNN definition)
// ============================================================

bool TrtRunner::inferBatchGather(const PendingNN* jobs, int B) {
    if (!ctx || B <= 0 || B > maxBatch) return false;

    for (int i = 0; i < B; ++i) {
        auto* dst = reinterpret_cast<NNInput*>(
            hInputPinned + (size_t)i * (size_t)NN_INPUT_SIZE
            );
        positionToNNInput(jobs[i].pos, *dst);
    }

#if AI_HAVE_CUDA_KERNELS
    for (int i = 0; i < B; ++i) {
        int* idxBase = hGatherIdxPinned + (size_t)i * (size_t)AI_MAX_MOVES;
        std::fill_n(idxBase, AI_MAX_MOVES, -1);

        const int n = jobs[i].ml.n;
        for (int j = 0; j < n; ++j) {
            const uint16_t k = jobs[i].policyIdx[(size_t)j];
            idxBase[j] = (k == INVALID_POLICY_IDX) ? -1 : (int)k;
        }
    }
#endif

    bool ok = runBatchAndSync(B);
    if (!ok) return false;

    for (int i = 0; i < B; ++i)
        hValuePinned[(size_t)i] = clamp01(hValuePinned[(size_t)i]);

    return true;
}

bool TrtRunner::inferBatchGather(const PendingNN* const* jobs, int B) {
    if (!ctx || B <= 0 || B > maxBatch) return false;

    for (int i = 0; i < B; ++i) {
        const PendingNN& job = *jobs[i];
        auto* dst = reinterpret_cast<NNInput*>(
            hInputPinned + (size_t)i * (size_t)NN_INPUT_SIZE
            );
        positionToNNInput(job.pos, *dst);
    }

#if AI_HAVE_CUDA_KERNELS
    for (int i = 0; i < B; ++i) {
        const PendingNN& job = *jobs[i];
        int* idxBase = hGatherIdxPinned + (size_t)i * (size_t)AI_MAX_MOVES;
        std::fill_n(idxBase, AI_MAX_MOVES, -1);

        const int n = job.ml.n;
        for (int j = 0; j < n; ++j) {
            const uint16_t k = job.policyIdx[(size_t)j];
            idxBase[j] = (k == INVALID_POLICY_IDX) ? -1 : (int)k;
        }
    }
#endif

    bool ok = runBatchAndSync(B);
    if (!ok) return false;

    for (int i = 0; i < B; ++i) {
        hValuePinned[(size_t)i] = clamp01(hValuePinned[(size_t)i]);
    }

    return true;
}

// ============================================================
// Inference server (single TensorRT owner thread)
// ============================================================

struct InferenceServer {
    static constexpr int NN_QUEUE_CAP = 8 * TRT_MAX_BATCH; // 2048 for TRT_MAX_BATCH=256

    MCTSTable& T;

    std::atomic<bool> stop{ false };
    std::atomic<int>  qSize{ 0 };

    std::mutex m;
    std::condition_variable cvNotEmpty;
    std::condition_variable cvNotFull;
    std::condition_variable cvIdle;

    std::deque<std::unique_ptr<PendingNN>> q;
    std::thread th;
    std::thread th2;

    int busyCount = 0; // protected by m; number of consumers processing a batch

    // Consumer runners: [0] is always g_trt; [1] is optional (pipelined mode).
    TrtRunner* runners[2] = { &g_trt, nullptr };
    int nRunners = 1;

    std::vector<float> neutralPol;     // [POLICY_SIZE]
    std::vector<float> neutralLogits;  // [AI_MAX_MOVES]

    explicit InferenceServer(MCTSTable& tab,
        TrtRunner* primaryRunner = nullptr,
        TrtRunner* secondRunner = nullptr) : T(tab) {
        q.clear();
        neutralPol.assign((size_t)POLICY_SIZE, 0.0f);
        neutralLogits.assign((size_t)AI_MAX_MOVES, 0.0f);
        if (primaryRunner) runners[0] = primaryRunner;
        if (secondRunner) {
            runners[1] = secondRunner;
            nRunners = 2;
        }
    }

    void start() {
        {
            std::lock_guard<std::mutex> lk(m);
            stop.store(false, std::memory_order_relaxed);
            busyCount = 0;
            q.clear();
            qSize.store(0, std::memory_order_relaxed);
        }
        th = std::thread([this] { this->run(0); });
        if (nRunners > 1) th2 = std::thread([this] { this->run(1); });
    }

    void stopAndDrain() {
        {
            std::lock_guard<std::mutex> lk(m);
            stop.store(true, std::memory_order_relaxed);
        }
        cvNotEmpty.notify_all();
        cvNotFull.notify_all();
        cvIdle.notify_all();

        if (th.joinable()) th.join();
        if (th2.joinable()) th2.join();
    }

    int size() const {
        return qSize.load(std::memory_order_relaxed);
    }

    void waitIdle() {
        std::unique_lock<std::mutex> lk(m);
        cvIdle.wait(lk, [&] {
            return q.empty() && busyCount == 0;
            });
    }

    bool submit(std::unique_ptr<PendingNN>&& job,
        const std::atomic<bool>* extCancel = nullptr,
        const std::atomic<bool>* extAbort = nullptr) {
        auto cancelled = [&]() -> bool {
            return stop.load(std::memory_order_relaxed) ||
                (extCancel && extCancel->load(std::memory_order_relaxed)) ||
                (extAbort && extAbort->load(std::memory_order_relaxed));
            };

        std::unique_lock<std::mutex> lk(m);

        while ((int)q.size() >= NN_QUEUE_CAP && !cancelled()) {
            cvNotFull.wait_for(lk, std::chrono::microseconds(AI_SUBMIT_WAIT_US));
        }

        if (cancelled()) return false;

        q.emplace_back(std::move(job));
        qSize.store((int)q.size(), std::memory_order_relaxed);

        lk.unlock();
        cvNotEmpty.notify_one();
        return true;
    }

private:
    int popBatchUnlocked(std::vector<std::unique_ptr<PendingNN>>& batch, int wantB) {
        batch.clear();
        batch.reserve((size_t)wantB);

        int n = 0;
        while (n < wantB && !q.empty()) {
            batch.emplace_back(std::move(q.front())); // FIFO
            q.pop_front();
            ++n;
        }

        qSize.store((int)q.size(), std::memory_order_relaxed);
        return n;
    }

    void run(int runnerIdx) {
        TrtRunner& R = *runners[runnerIdx];

        std::vector<std::unique_ptr<PendingNN>> batch;
        std::vector<std::unique_ptr<PendingNN>> add;
        std::vector<const PendingNN*> batchPtrs;
        batch.reserve((size_t)TRT_MAX_BATCH);
        add.reserve((size_t)TRT_MAX_BATCH);

        auto processBatch = [&](std::vector<std::unique_ptr<PendingNN>>& jobs) {
            const int B = (int)jobs.size();
            if (B <= 0) return;

            batchPtrs.resize((size_t)B);
            for (int i = 0; i < B; ++i) batchPtrs[(size_t)i] = jobs[(size_t)i].get();

#if AI_HAVE_CUDA_KERNELS
            bool ok = R.inferBatchGather(batchPtrs.data(), B);
            for (int i = 0; i < B; ++i) {
                float v = ok ? R.valueHost(i) : 0.5f;
                const float* logits = ok ? R.gatherLogitsHostPtr(i) : neutralLogits.data();
                expandLeafWithGatheredLogits(T, *jobs[(size_t)i], v, logits);
            }
#else
            std::vector<Position> posBatch((size_t)B);
            for (int i = 0; i < B; ++i) posBatch[(size_t)i] = jobs[(size_t)i]->pos;

            bool ok = R.inferBatch(posBatch.data(), B);
            for (int i = 0; i < B; ++i) {
                float v = ok ? R.valueHost(i) : 0.5f;
                const float* pol = ok ? R.policyHostPtr(i) : neutralPol.data();
                expandLeafWithOutputs(T, *jobs[(size_t)i], v, pol);
            }
#endif
            };

        for (;;) {
            {
                std::unique_lock<std::mutex> lk(m);

                if (q.empty() && busyCount == 0) cvIdle.notify_all();

                cvNotEmpty.wait(lk, [&] {
                    return stop.load(std::memory_order_relaxed) || !q.empty();
                    });

                if (stop.load(std::memory_order_relaxed) && q.empty()) break;

                ++busyCount;
                (void)popBatchUnlocked(batch, TRT_MAX_BATCH);
            }

            // queue shrank -> wake blocked producers
            cvNotFull.notify_all();

            // small fill window to improve batch utilization
            const auto tFillEnd =
                std::chrono::steady_clock::now() + std::chrono::microseconds(200);

            while ((int)batch.size() < TRT_MAX_BATCH &&
                std::chrono::steady_clock::now() < tFillEnd) {
                std::unique_lock<std::mutex> lk(m);

                if (q.empty()) {
                    cvNotEmpty.wait_until(lk, tFillEnd, [&] {
                        return stop.load(std::memory_order_relaxed) || !q.empty();
                        });
                }

                if (q.empty()) break;

                add.clear();
                const int need = TRT_MAX_BATCH - (int)batch.size();
                (void)popBatchUnlocked(add, need);
                lk.unlock();

                cvNotFull.notify_all();

                for (auto& j : add) batch.emplace_back(std::move(j));
            }

            processBatch(batch);
            freePendingBatch(batch);

            {
                std::lock_guard<std::mutex> lk(m);
                --busyCount;
                if (q.empty() && busyCount == 0) cvIdle.notify_all();
            }
        }

        {
            std::lock_guard<std::mutex> lk(m);
            qSize.store((int)q.size(), std::memory_order_relaxed);
            if (q.empty() && busyCount == 0) cvIdle.notify_all();
        }

        cvNotFull.notify_all();
    }
};
inline InferenceServerStopGuard::~InferenceServerStopGuard() noexcept {
    if (!srv) return;
    try {
        srv->stopAndDrain();
    }
    catch (...) {
    }
}
static void ensureRootExpanded(MCTSTable& T,
    const Position& rootPos,
    const std::array<uint64_t, 4>& path,
    const std::array<int, 64>& mask,
    MoveList& ml,
    int term,
    TrtRunner& RT = g_trt) {
    TTNode* root = T.getNode(rootPos.key);
    if (!root) return;

    uint8_t ex = root->expanded.load(std::memory_order_acquire);
    if (ex == 1) return;
    if (ex == 2) { waitWhileExpanding(root); return; }

    uint8_t expected = 0;
    if (!root->expanded.compare_exchange_strong(expected, 2,
        std::memory_order_acq_rel,
        std::memory_order_relaxed)) {
        return;
    }
    ExpansionClaimGuard rootClaim(root);

    if (term) {
        root->key = rootPos.key;
        root->edgeBegin = 0;
        root->edgeCount = 0;
        root->terminal = 1;
        root->chance = 0;

        Trace empty; empty.reset();
        backprop(root, 1.0f, empty);
        publishTerminalWithMove(T, root, rootPos.key, ml.n ? ml.m[0] : 0);
        rootClaim.release();
        return;
    }

    if (ml.n == 0) {
        publishReady(root, rootPos.key, 0, 0, 0, 1);
        rootClaim.release();
        return;
    }

    PendingNN p;
    resetPendingNN(p);
    PendingNNGuard pGuard(p);

    p.leaf = root;
    p.pos = rootPos;
    p.ml = ml;
    p.trace.reset();
    fillPendingPolicyIdx(p);

    rootClaim.release(); // cleanup of expansion now owned by pGuard / p
    float v = 0.5f;

#if AI_HAVE_CUDA_KERNELS
    if (!RT.inferBatchGather(&p, 1)) {
        v = 0.5f;
        std::vector<float> z((size_t)AI_MAX_MOVES, 0.0f);
        expandLeafWithGatheredLogits(T, p, v, z.data());
        pGuard.release();
        return;
    }
    v = RT.valueHost(0);
    const float* logits = RT.gatherLogitsHostPtr(0);
    expandLeafWithGatheredLogits(T, p, v, logits);
    pGuard.release();
#else
    std::vector<float> pol((size_t)POLICY_SIZE, 0.0f);
    if (!RT.inferBatch(&p.pos, 1, &v, pol.data())) {
        v = 0.5f;
        std::fill(pol.begin(), pol.end(), 0.0f);
    }
    expandLeafWithOutputs(T, p, v, pol.data());
    pGuard.release();
#endif
}

struct SimDiag {
    uint32_t ttHit = 0;
    uint32_t ttMiss = 0;
    uint32_t depth = 0;
};

static std::atomic<uint64_t> g_failGetNode{ 0 };
static std::atomic<uint64_t> g_failExpandWait{ 0 };
static std::atomic<uint64_t> g_failDepth{ 0 };

static bool runOneSim(MCTSTable& T,
    const Position& rootPos,
    const std::array<uint64_t, 4>& path,
    const std::array<int, 64>& mask,
    PendingNN& outPending,
    bool& outNeedNN,
    uint32_t rngJitter,
    const SearchParams& searchParams,
    SimDiag* diag = nullptr) {

    outNeedNN = false;

    if (AI_UNLIKELY(T.abort.load(std::memory_order_relaxed))) {
        return false;
    }

    Position pos = rootPos;
    Trace tr; tr.reset();
    bool isRoot = true;
    uint32_t decisionDepth = 0;

    for (;;) {
        if (AI_UNLIKELY(T.abort.load(std::memory_order_relaxed))) {
            rollbackVirtualLoss(tr);
            return false;
        }

        // Depth guard
        if (tr.n >= MCTS_MAX_DEPTH - 2) {
            g_failDepth.fetch_add(1, std::memory_order_relaxed);
            rollbackVirtualLoss(tr);
            return false;
        }

        TTNode* node = T.getNode(pos.key);
        if (!node) {
            g_failGetNode.fetch_add(1, std::memory_order_relaxed);
            rollbackVirtualLoss(tr);
            return false;
        }

        uint8_t ex = node->expanded.load(std::memory_order_acquire);

        if (diag) {
            if (ex == 0) ++diag->ttMiss;
            else         ++diag->ttHit;
        }

        // Someone else expanding
        if (ex == 2) {
            if (!waitWhileExpanding(node)) {
                g_failExpandWait.fetch_add(1, std::memory_order_relaxed);
                rollbackVirtualLoss(tr);
                return false;
            }
            continue;
        }

        // Need expansion
        if (ex == 0) {
            uint8_t expected = 0;
            if (!node->expanded.compare_exchange_strong(expected, 2,
                std::memory_order_acq_rel,
                std::memory_order_relaxed)) {
                continue;
            }
            ExpansionClaimGuard claim(node);

            MoveList ml;
            int term = 0;
            Position tmp = pos;

            genLegal(tmp,
                path,
                mask,
                ml, term);

            if (term) {
                node->key = pos.key;
                node->edgeBegin = 0;
                node->edgeCount = 0;
                node->terminal = 1;
                node->chance = 0;

                backprop(node, 1.0f, tr, &T);
                publishTerminalWithMove(T, node, pos.key, ml.n ? ml.m[0] : 0);
                claim.release();
                if (diag) diag->depth = decisionDepth + 1;
                return true;
            }

            if (ml.n == 0) {
                // Chance node
                publishReady(node, pos.key, 0, 0, 0, 1);
                claim.release();

                tr.push(node, nullptr, /*flip=*/true, /*vloss=*/false);
                makeRandomDeterministic(pos, node);
                isRoot = false;
                continue;
            }

            // Need NN
            outNeedNN = true;
            outPending.leaf = node;
            outPending.pos = pos;
            outPending.ml = ml;
            outPending.trace = tr;
            fillPendingPolicyIdx(outPending);

            claim.release(); // ownership of expansion cleanup transferred to PendingNN
            if (diag) diag->depth = decisionDepth;
            return true;
        }

        // Expanded
        if (node->terminal) {
            backprop(node, 1.0f, tr, &T);
            if (diag) diag->depth = decisionDepth + 1;
            return true;
        }

        if (node->edgeCount == 0) {
            if (node->chance) {
                tr.push(node, nullptr, /*flip=*/true, /*vloss=*/false);
                makeRandomDeterministic(pos, node);
                isRoot = false;
                continue;
            }
            else {
                float vLeaf = nodeQ(*node);
                backprop(node, vLeaf, tr, &T);
                if (diag) diag->depth = decisionDepth;
                return true;
            }
        }

        // Decision node: PUCT
        const uint32_t pv = node->visits.load(std::memory_order_relaxed);
        const float parentQ = nodeQ(*node);
        const float cpuct = cpuctFromVisits(pv, isRoot, searchParams);

        TTEdge* e0 = T.edgePtr(node->edgeBegin);
        int bestI = selectPUCT(*node, e0, cpuct, pv, parentQ, searchParams, rngJitter);
        TTEdge* e = &e0[bestI];

        // Classic virtual loss (mark the selected edge as "in flight")
        TraceStep& step = tr.push(node, e, /*flip=*/false, /*vloss=*/true);
        if (AI_UNLIKELY(!applyVirtualLoss(step))) {
            step.vloss = false;
            T.abort.store(true, std::memory_order_release);
            return false;
        }

        makeMove(pos, mask, e->move);
        ++decisionDepth;
        isRoot = false;
    }
}
static AI_FORCEINLINE char promoChar(int promo) {
    switch (promo) {
    case 1: return 'n';
    case 2: return 'b';
    case 3: return 'r';
    case 4: return 'q';
    default: return 0;
    }
}

static std::string moveToStr(int move) {
    int from = move & 63;
    int to = (move >> 6) & 63;
    int promo = (move >> 12) & 7;

    std::string s = sqName(from) + sqName(to);
    char pc = promoChar(promo);
    if (pc) s.push_back(pc);
    return s;
}
void extractBestPVUntilChance(MCTSTable& T, Position& rootPos, array<int, 64>& mask, vector<int>& outPV, uint64_t& key) {
    outPV.clear();
    Position pos = rootPos;
    while (1) {
        TTNode* n = T.findNodeNoInsert(pos.key);
        if (!n || n->expanded.load(memory_order_acquire) != 1 || n->edgeCount == 0) {
            key = pos.key;
            return;
        }
        TTEdge* e0 = T.edgePtr(n->edgeBegin);
        if (n->terminal) {
            outPV.push_back(e0[0].move);
            key = 0;
            return;
        }
        int m = e0[selectBestPVEdge(*n, e0)].move;
        outPV.push_back(m);
        makeMove(pos, mask, m);
    }
}
uint64_t terminalAwareKeyAfterLine(MCTSTable& T, Position& rootPos, array<int, 64>& mask, vector<int>& line) {
    Position pos = rootPos;
    for (int m : line) {
        TTNode* n = T.findNodeNoInsert(pos.key);
        if (!n || n->expanded.load(memory_order_acquire) != 1)return pos.key;
        if (n->terminal)return 0;
        bool found = false;
        TTEdge* e0 = T.edgePtr(n->edgeBegin);
        for (int i = 0; i < n->edgeCount; i++)if (e0[i].move == m) {
            found = true;
            break;
        }
        if (!found)return pos.key;
        makeMove(pos, mask, m);
    }
    TTNode* n = T.findNodeNoInsert(pos.key);
    if (!n || n->expanded.load(memory_order_acquire) != 1 || !n->terminal)return pos.key;
    return 0;
}
int Alternative(moveState& cur, moveState& alt, MCTSTable& T, Position& rootPos, array<int, 64>& mask) {
    if (alt.pvKey == cur.pvKey)return 0;
    vector<int> line;
    line.push_back(cur.move);
    for (int m : alt.pv)if (m != cur.move)line.push_back(m);
    return terminalAwareKeyAfterLine(T, rootPos, mask, line) != alt.pvKey;
}
double computeDifForRootMoves(int write, vector<moveState>& rootMoves, MCTSTable& T, Position& rootPos, array<int, 64>& mask) {
    auto toSidePerspective = [&rootPos](double eval) {return rootPos.side == 0 ? eval : 1 - eval; };
    if (rootMoves.empty())return 100;
    stable_sort(rootMoves.begin(), rootMoves.end(), [](const moveState& a, const moveState& b) {return a.pvKey == 0 && b.pvKey; });
    if (rootMoves[0].visits == 0 || rootMoves[0].pvKey == 0) {
        for (moveState& ms : rootMoves)ms.dif = -100;
        rootMoves[0].dif = 100;
        return 100;
    }
    for (moveState& ms : rootMoves) {
        ms.dif = -100;
        if (Alternative(ms, rootMoves[0], T, rootPos, mask) || write == 0 && ms.pvKey != rootMoves[0].pvKey) {
            ms.dif = 100;
            continue;
        }
        for (moveState& alt : rootMoves)if (Alternative(ms, alt, T, rootPos, mask) && alt.visits)ms.dif = max(ms.dif, toSidePerspective(alt.eval));
    }
    stable_sort(rootMoves.begin(), rootMoves.end(), [](const moveState& a, const moveState& b) {return a.dif < b.dif; });
    for (moveState& ms : rootMoves)if (ms.dif >= 0 && ms.dif <= 1)ms.dif = 100 * (toSidePerspective(ms.eval) - ms.dif); else ms.dif *= -1;
    return rootMoves[0].dif;
}
void Dif(double dif) { if (dif > -100)cout << showpos << setprecision(2) << "dif=" << dif << noshowpos << setprecision(6); }
void collectRootMoves(MCTSTable& T, const Position& rootPos, float& outQSideToMove, vector<moveState>& outMoves);
void extractDifPVUntilChance(int write, MCTSTable& T, Position& rootPos, array<int, 64>& mask, vector<moveState>& rootMoves, vector<int>& outPV) {
    outPV.clear();
    Position pos = rootPos;
    if (rootMoves.empty())return;
    outPV.push_back(rootMoves[0].move);
    makeMove(pos, mask, rootMoves[0].move);
    while (1) {
        TTNode* n = T.findNodeNoInsert(pos.key);
        if (!n || n->expanded.load(memory_order_acquire) != 1 || n->edgeCount == 0)return;
        float q;
        vector<moveState> moves;
        collectRootMoves(T, pos, q, moves);
        if (n->terminal) {
            outPV.push_back(moves[0].move);
            return;
        }
        for (moveState& ms : moves) {
            if (pos.side && ms.eval >= 0)ms.eval = 1 - ms.eval;
            Position p = pos;
            makeMove(p, mask, ms.move);
            extractBestPVUntilChance(T, p, mask, ms.pv, ms.pvKey);
            ms.pv.insert(ms.pv.begin(), ms.move);
        }
        computeDifForRootMoves(write, moves, T, pos, mask);
        outPV.push_back(moves[0].move);
        makeMove(pos, mask, moves[0].move);
    }
}
int getMaxVisitsLen(vector<moveState>& rootMoves) {
    int maxLen = 0;
    for (moveState& ms : rootMoves)maxLen = max(maxLen, int(to_string(ms.visits).size()));
    return maxLen;
}
mutex posMutex;
int ROLL;
Position POS;
array<uint64_t, 4> PATH;
array<int, 64> MASK;
// Number of leaf-producing worker threads for timed search.
// Capped at 2: producers busy-spin in the queue throttle, and on a
// power-limited laptop extra spinning cores steal thermal budget from the GPU
// (measured: 6 producers scored worse than 1 at 0.4s/move).
static unsigned autoSearchThreads() {
    unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    unsigned t = (hw > 2u) ? (hw - 2u) : 1u;
    return std::min(t, 2u);
}
// A tree only has to hold one turn's worth of search: after the dice are thrown
// the position changes at random and everything below is about other rolls.
// Sizing it to the allowance keeps the edge pool from filling up mid-turn and
// keeps the table small enough to stay cache-friendly.
static void tableSizeForTime(double sec, size_t& nodes, size_t& edges) {
    const double sims = std::max(0.25, sec) * 35000.0 * 2.0;   // measured rate, doubled
    size_t p = 1ull << 16;
    while ((double)p < sims) p <<= 1;
    nodes = p;
    edges = p * 32;
}

// How often the adaptive budget cut a search short, and why.
static std::atomic<long long> g_adaptStopSettled{ 0 }, g_adaptStopFree{ 0 }, g_adaptSearches{ 0 };
static double g_gapSum = 0.0, g_wantSum = 0.0;
static long long g_gapN = 0;
// Correction factor per side, nudged after every move so that the average spend
// settles on the allowance. It has to be integral: setting it to 1/spend has a
// fixed point at the square root of the target, not at the target.
static double g_creditP1 = 1.0, g_creditP2 = 1.0;
static double g_probeSec = 0.0;        // time the adaptive probe itself costs
static long long g_probeN = 0;
static std::vector<double> g_spreadSamples(200000);
static size_t g_spreadN = 0;

// Follows the most-visited moves from p and reports where the turn ends: a
// terminal position, or one where the dice are thrown again. Returns false if
// the tree has not been built that far - such a line says nothing about the
// outcome of the turn and must not be compared against another.
static bool turnEndKey(MCTSTable& T, Position p,
    const std::array<int, 64>& mask, uint64_t& outKey) {
    for (int guard = 0; guard < 8; ++guard) {
        TTNode* n = T.findNodeNoInsert(p.key);
        if (!n || n->expanded.load(std::memory_order_acquire) != 1) return false;
        if (n->terminal || n->chance || n->edgeCount == 0) { outKey = p.key; return true; }
        TTEdge* e0 = T.edgePtr(n->edgeBegin);
        makeMove(p, mask, e0[selectBestPVEdge(*n, e0)].move);
    }
    return false;
}

// Every position the turn can end in, gathered from the tree rather than from
// one greedy line per root move: inside a turn the moves commute, so a single
// root move leads to many possible endings and the best alternative may well be
// down a line that is not that move's principal variation.
struct TurnEnd { uint64_t key; uint64_t visits; double eval; double sum; };

static void collectTurnEnds(MCTSTable& T, Position p,
    const std::array<int, 64>& mask,
    std::vector<TurnEnd>& out, int depth, int& budget) {
    if (--budget < 0) return;
    TTNode* n = T.findNodeNoInsert(p.key);
    // Everything needed is already in the node: expansion ran genLegal once and
    // stored both the edges and whether this is where the turn ends.
    if (!n || n->expanded.load(std::memory_order_acquire) != 1) return;
    if (n->terminal || n->chance || n->edgeCount == 0) {
        const uint64_t v = n->visits.load(std::memory_order_relaxed);
        if (!v) return;
        for (auto& e : out) if (e.key == p.key) return;   // same position, already taken
        out.push_back({ p.key, v, (double)nodeQ(*n),
                        n->valueSum.load(std::memory_order_relaxed) });
        return;
    }
    if (depth <= 0) return;
    TTEdge* e0 = T.edgePtr(n->edgeBegin);
    for (int i = 0; i < (int)n->edgeCount; ++i) {
        if (!e0[i].visits.load(std::memory_order_relaxed)) continue;
        Position q = p;
        makeMove(q, mask, e0[i].move);
        collectTurnEnds(T, q, mask, out, depth - 1, budget);
    }
}

// First move of a sequence that takes the enemy king without giving the turn
// away, read off the tree the search has already built. Moves inside a turn do
// not change the side, so everything above the first chance node belongs to us.
static int findWinInTree(MCTSTable& T, const Position& pos,
    const std::array<int, 64>& mask, int depth) {
    TTNode* n = T.findNodeNoInsert(pos.key);
    if (!n || n->expanded.load(std::memory_order_acquire) != 1 || n->edgeCount == 0) return 0;
    if (n->chance) return 0;
    TTEdge* e0 = T.edgePtr(n->edgeBegin);
    if (n->terminal) return e0[0].move;
    if (depth <= 1) return 0;
    for (int i = 0; i < (int)n->edgeCount; ++i) {
        Position q = pos;
        makeMove(q, mask, e0[i].move);
        if (findWinInTree(T, q, mask, depth - 1)) return e0[i].move;
    }
    return 0;
}

void mctsBatchedMT(MCTSTable& T,
    Position& rootPos,
    std::array<uint64_t, 4>& path,
    std::array<int, 64>& mask,
    double timeSec,
    float& outEvalWhite,
    float& outAvgDepth,
    std::vector<moveState>& outRootMoves,
    std::vector<int>& outPVBeforeRoll,
    int write,
    int abort,
    unsigned searchThreads = 1,
    bool dualInfer = false,
    TrtRunner* primaryRunner = nullptr,   // nullptr => g_trt
    InferenceServer* extServer = nullptr, // persistent server (skips per-move start/stop)
    bool stopOnWin = false,               // return as soon as a win is in the tree
    int adaptMode = 0,                    // 1 = settled leader, 2 = final-position gap
    double adaptBaseSec = 0.0,            // the move's nominal allowance
    const std::atomic<bool>* externalStop = nullptr, // stop the moment this is set
    double adaptCredit = 0.0)             // unspent budget, in units of the allowance
{
    MoveList ml;
    int term;
    genLegal(rootPos, path, mask, ml, term);

    outPVBeforeRoll.clear();

    if (term) {
        outEvalWhite = 1 - rootPos.side;
        outAvgDepth = 1.0f;
        outRootMoves.clear();
        outRootMoves.push_back({ ml.m[0], outEvalWhite, 0, 0.0f, 0ull });
        outPVBeforeRoll.push_back(ml.m[0]);
        if (write == 2) {
            clearConsoleFull();
            std::cout << moveToStr(ml.m[0]) << std::endl;
        }
        return;
    }

    TTNode* rootNode = T.getNode(rootPos.key);
    if (!rootNode) {
        outEvalWhite = 0.5f;
        outAvgDepth = 0.0f;
        outRootMoves.clear();
        outPVBeforeRoll.clear();
        return;
    }

    TrtRunner& RT = primaryRunner ? *primaryRunner : g_trt;
    ensureRootExpanded(T, rootPos, path, mask, ml, term, RT);

    if (T.abort.load(std::memory_order_acquire)) {
        outEvalWhite = 0.5f;
        outAvgDepth = 0.0f;
        outRootMoves.clear();
        outPVBeforeRoll.clear();
        return;
    }

    // The dual (pipelined) runner mirrors g_trt weights; never pair it with a custom primary.
    const bool useDual = dualInfer && g_trt2Ready && (!primaryRunner || primaryRunner == &g_trt);

    InferenceServer localServer(T, &RT, useDual ? &g_trt2 : nullptr);
    InferenceServer& nnServer = extServer ? *extServer : localServer;
    if (!extServer) localServer.start();
    InferenceServerStopGuard nnServerGuard(localServer);
    if (extServer) nnServerGuard.release(); // local server unused; do not touch the external one

    const unsigned threads = std::max(1u, searchThreads);

    const auto t0 = std::chrono::steady_clock::now();
    const auto tEnd = t0 + std::chrono::duration<double>(timeSec);
    auto tNextWrite = t0 + std::chrono::seconds(1);

    std::atomic<bool> stop{ false };
    AtomicStopGuard stopGuard(stop);

    std::atomic<uint64_t> simOK{ 0 }, simFail{ 0 }, nnExp{ 0 }, depthSum{ 0 };

    auto worker = [&](unsigned tid) {
        uint32_t jitterBase = (uint32_t)(0x9E3779B9u * (tid + 1));

        uint64_t iter = 0;
        int queueSpins = 0;

        for (;;) {
            if (stop.load(std::memory_order_relaxed)) break;
            if (T.abort.load(std::memory_order_relaxed)) break;

            if ((iter++ & 63ull) == 0ull) {
                if (std::chrono::steady_clock::now() >= tEnd) break;
            }

            bool didUsefulWork = false;

            const int B = std::max(1, g_nnBatch);
            for (int k = 0; k < B; ++k) {
                if (T.abort.load(std::memory_order_relaxed)) break;
                if (stop.load(std::memory_order_relaxed)) break;

                // Front-pressure: let NN server drain before making more leaves.
                throttleOnNNQueue_NoSleep(nnServer.size(), queueSpins);

                if (T.abort.load(std::memory_order_relaxed)) break;
                if (stop.load(std::memory_order_relaxed)) break;

                PendingNN localPending;
                resetPendingNN(localPending);

                bool needNN = false;

                SimDiag sd{};
                bool ok = runOneSim(T, rootPos, path, mask,
                    localPending, needNN,
                    jitterBase + (uint32_t)k * 1337u,
                    kDefaultSearchParams, &sd);

                if (!ok) {
                    simFail.fetch_add(1, std::memory_order_relaxed);
                    if (T.abort.load(std::memory_order_relaxed)) break;
                    continue;
                }

                didUsefulWork = true;
                simOK.fetch_add(1, std::memory_order_relaxed);
                depthSum.fetch_add(sd.depth, std::memory_order_relaxed);

                if (needNN) {
                    nnExp.fetch_add(1, std::memory_order_relaxed);

                    // Extra throttle immediately before submit.
                    throttleOnNNQueue_NoSleep(nnServer.size(), queueSpins);

                    if (stop.load(std::memory_order_relaxed) ||
                        T.abort.load(std::memory_order_relaxed)) {
                        cancelPendingNN(localPending);
                        simFail.fetch_add(1, std::memory_order_relaxed);
                        break;
                    }

                    auto p = allocPendingNN();
                    *p = localPending; // copy only when NN submit is really needed

                    if (!nnServer.submit(std::move(p), &stop, &T.abort)) {
                        if (p) {
                            cancelPendingNN(*p);
                            freePendingNN(std::move(p));
                        }
                        simFail.fetch_add(1, std::memory_order_relaxed);
                        if (T.abort.load(std::memory_order_relaxed)) break;
                        continue;
                    }
                }
            }

            if (!didUsefulWork) {
                cpuRelax();
            }
        }
        };

    std::vector<std::thread> pool;
    pool.reserve(threads);
    ThreadJoinGuard poolGuard(pool);

    for (unsigned t = 0; t < threads; ++t) {
        pool.emplace_back(worker, t);
    }

    auto emitSearchSnapshot = [&]() {
        float qRootNow = nodeQ(*rootNode);
        float mctsEvalWhiteNow = (rootPos.side == 0) ? qRootNow : (1.0f - qRootNow);
        const uint64_t simsNow = simOK.load(std::memory_order_relaxed);
        const uint64_t depthNow = depthSum.load(std::memory_order_relaxed);
        const double avgDepthNow = simsNow ? (double)depthNow / (double)simsNow : 0.0;

        std::vector<moveState> rootMovesNow;
        uint8_t exNow = rootNode->expanded.load(std::memory_order_acquire);
        if (exNow == 1 && rootNode->edgeCount) {
            TTEdge* e0 = T.edgePtr(rootNode->edgeBegin);
            rootMovesNow.reserve(rootNode->edgeCount);
            for (int i = 0; i < (int)rootNode->edgeCount; ++i) {
                const TTEdge& e = e0[i];
                uint32_t v = e.visits.load(std::memory_order_relaxed);
                float p = e.prior();
                float ev = -1.0f;
                if (v) ev = clamp01(e.sum() / (float)v);
                rootMovesNow.push_back(moveState{ e.move, ev, v, p, 0ull, {} });
            }
            std::sort(rootMovesNow.begin(), rootMovesNow.end(),
                [](const moveState& a, const moveState& b) {
                    if (a.visits != b.visits) return a.visits > b.visits;
                    return a.eval > b.eval;
                });
            if (rootPos.side == 1) {
                for (auto& ms : rootMovesNow) {
                    if (ms.eval >= 0.0f) ms.eval = 1.0f - ms.eval;
                }
            }
            for (auto& ms : rootMovesNow) {
                Position p = rootPos;
                makeMove(p, mask, ms.move);
                extractBestPVUntilChance(T, p, mask, ms.pv, ms.pvKey);
                ms.pv.insert(ms.pv.begin(), ms.move);
            }
        }

        std::vector<int> pvNow;
        double dif = computeDifForRootMoves(write, rootMovesNow, T, rootPos, mask);
        extractDifPVUntilChance(write, T, rootPos, mask, rootMovesNow, pvNow);

        clearConsoleFull();
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "depth=" << avgDepthNow << '\n';
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "eval=" << mctsEvalWhiteNow << '\n';
        for (size_t i = 0; i < pvNow.size(); ++i) {
            if (i) std::cout << ' ';
            std::cout << moveToStr(pvNow[i]);
        }

        cout << endl;
        Dif(dif);

        std::cout << '\n';
        for (const auto& ms : rootMovesNow) {
            int d = (int)std::to_string(ms.visits).size();
            int spacesBeforePrior = 1 + (getMaxVisitsLen(rootMovesNow) - d);

            std::cout
                << moveToStr(ms.move)
                << " eval " << ms.eval
                << " visits " << ms.visits
                << std::string(spacesBeforePrior, ' ')
                << "prior " << ms.prior
                << ' ';
            Dif(ms.dif);
            cout << '\n';
        }
        std::cout.flush();
        };

    bool forceExit = false;
    int winMove = 0;
    int winPoll = 0;
    int adaptPoll = 0, adaptStable = 0, adaptLeader = 0;
    std::vector<TurnEnd> prevEnds;        // previous snapshot, for the churn test
    float qIgnored = 0.0f;
    const auto tStart = std::chrono::steady_clock::now();
    if (adaptMode) ++g_adaptSearches;
    while (std::chrono::steady_clock::now() < tEnd) {
        if (T.abort.load(std::memory_order_relaxed)) break;
        // Filling the time the mouse needs: the search runs until the move on
        // the board is confirmed, no longer and no shorter.
        if (externalStop && externalStop->load(std::memory_order_relaxed)) break;
        auto now = std::chrono::steady_clock::now();

        if (abort && POS.key != rootPos.key) {
            forceExit = true;
            break;
        }

        // Once the tree contains a way to take the king this turn there is
        // nothing left to weigh up, so stop right there.
        if (stopOnWin && ++winPoll % 15 == 0) {
            winMove = findWinInTree(T, rootPos, mask, 3);
            if (winMove) break;
        }

        // Adaptive budget. Two things end the search early: a move that cuts no
        // reachable position off (playing it postpones the decision for free),
        // and a leader that has stopped changing. Time is worth spending on the
        // probability of being wrong, not on the size of the gap - two equal
        // lines cost nothing to confuse.
        // Mode 7 is a control, not a policy: it alternates short and long
        // searches by position parity, with the same average. If it loses too,
        // then uneven spending is being punished by itself - short searches pay
        // ramp-up and warm-up costs twice - and no amount of cleverness in
        // choosing where to be short can win.
        if (adaptMode == 7 && ++adaptPoll % 12 == 0) {
            const double elapsed = std::chrono::duration<double>(now - tStart).count();
            const double want = adaptBaseSec * ((rootPos.key & 1ull) ? 1.5 : 0.5);
            if (elapsed >= want) { ++g_adaptStopSettled; break; }
        }

        // Mode 5 asks a different question from the others: not "is the choice
        // obvious" but "is this position worth thinking about at all". A game
        // that is already decided cannot be improved by more search, while a
        // level, sharp one is exactly where the time belongs. Unlike the other
        // rules this one mostly ADDS time, which matters because the tree is
        // reused across the positions of a turn - cutting the first search
        // short degrades every search after it.
        if (adaptMode == 5 && ++adaptPoll % 12 == 0) {
            const double elapsed = std::chrono::duration<double>(now - tStart).count();
            const uint32_t rv = rootNode->visits.load(std::memory_order_relaxed);
            if (rv > 64) {
                const double q = (double)nodeQ(*rootNode);      // from our side
                const double decided = std::fabs(q - 0.5) * 2.0;   // 0 level, 1 settled
                // Linear, not squared: measured mean "decidedness" is only 0.25,
                // so a squared term leaves almost every position looking equally
                // sharp and the rule stops discriminating. The factor is set so
                // that a typical position draws its nominal allowance.
                double want = adaptBaseSec * 1.17 * (1.0 - decided);
                want = std::max(want, adaptBaseSec * 0.3);
                if (adaptCredit > 0.0) want *= adaptCredit;
                g_gapSum += decided; g_wantSum += want / std::max(1e-9, adaptBaseSec); ++g_gapN;
                if (elapsed >= want) { ++g_adaptStopSettled; break; }
            }
        }

        if ((adaptMode == 2 || adaptMode == 3 || adaptMode == 4 || adaptMode == 6 || adaptMode == 8)
            && ++adaptPoll % 12 == 0) {
            const double elapsed =
                std::chrono::duration<double>(now - tStart).count();
            if (elapsed >= adaptBaseSec * 0.2 &&
                rootNode->expanded.load(std::memory_order_acquire) == 1 &&
                rootNode->edgeCount) {
                // Group the root moves by the position the turn ends in: inside
                // a turn the moves commute, so what is really being chosen is a
                // final position, not a first move. The gap between the best
                // reachable final position and the next different one says how
                // much the decision is worth - a wide gap means the choice is
                // already made and thinking on is waste.
                {
                    const auto tProbe = std::chrono::steady_clock::now();
                    ++g_probeN;
                    std::vector<TurnEnd> fin;
                    int budget = 1200;
                    collectTurnEnds(T, rootPos, mask, fin, 6, budget);
                    // Where the line we would actually play ends up. The PV is
                    // greedy per step, and a final position's visits are the sum
                    // over every order of moves that reaches it, so the end of
                    // the PV is not necessarily the most visited position.
                    uint64_t pvEnd = 0;
                    const bool pvResolved = turnEndKey(T, rootPos, mask, pvEnd);
                    size_t mineIdx = fin.size();
                    if (pvResolved)
                        for (size_t i = 0; i < fin.size(); ++i) if (fin[i].key == pvEnd) { mineIdx = i; break; }

                    // Think long whenever the comparison cannot be made: either
                    // the end of the PV is not resolved, or nothing resolved
                    // differs from it. Cutting the search short on either would
                    // be guessing - but "long" still has to answer to the
                    // budget, or a short control where nothing ever resolves
                    // silently doubles this side's time.
                    // Wide enough not to clip the proportionality itself: with
                    // importance spanning about fourfold, time has to span the
                    // same range or the rule stops being t proportional to w.
                    const double ceiling = adaptBaseSec * 3.0 *
                        (adaptCredit > 0.0 ? adaptCredit : 1.0);
                    if (!(pvResolved && mineIdx < fin.size() && fin.size() >= 2)) {
                        if (elapsed >= ceiling) {
                            ++g_adaptStopFree;
                            g_probeSec += std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - tProbe).count();
                            break;
                        }
                    }
                    else {
                        // What the line is worth is decided by the best position
                        // it rules out, not by the next most popular one.
                        // Evaluations here come from the end positions' own
                        // nodes, and inside a turn the side to move has not
                        // changed, so they already read from our side.
                        // An outcome seen five times is not evidence the way one
                        // seen five thousand times is, so each evaluation is
                        // pulled towards the root's until its own visits earn it
                        // the right to stand on its own.
                        const double qRoot = (double)nodeQ(*rootNode);
                        const double K = 24.0;
                        auto trusted = [&](const TurnEnd& e) {
                            return (e.eval * (double)e.visits + qRoot * K) / ((double)e.visits + K);
                            };
                        double bestAlt = -1.0;
                        uint64_t altVisits = 1;
                        for (size_t i = 0; i < fin.size(); ++i) {
                            if (i == mineIdx) continue;
                            const double e = trusted(fin[i]);
                            if (e > bestAlt) {
                                bestAlt = e;
                                altVisits = std::max<uint64_t>(1, fin[i].visits);
                            }
                        }
                        const double gap = std::max(0.0, trusted(fin[mineIdx]) - bestAlt);
                        const uint64_t mineVisits = std::max<uint64_t>(1, fin[mineIdx].visits);
                        // Time allowed shrinks as the gap widens. The pivot is
                        // the measured median gap, so a typical position gets
                        // its nominal allowance and the budget is redistributed
                        // rather than merely trimmed.
                        // Pivot measured in play: comparing the line we would
                        // play against the best position it rules out, the
                        // typical gap is about 0.003, not the 0.03 seen when
                        // comparing two arbitrary root moves.
                        // Time is worth spending where the line we would play is
                        // barely ahead of the best line it rules out. The pivot
                        // only sets the shape; the level is held by the caller's
                        // correction, so the average lands on the allowance
                        // instead of drifting under it.
                        double want = adaptBaseSec * (0.004 / std::max(gap, 0.0004));
                        if (adaptMode == 6) {
                            // What an error really costs: how far apart the
                            // reachable outcomes are, weighted by how likely the
                            // search is to pick each. Comparing only the top two
                            // misses this - MCTS levels the top, so that gap is
                            // nearly constant and says almost nothing.
                            double wsum = 0.0, mean = 0.0;
                            for (const TurnEnd& e : fin) { wsum += (double)e.visits; }
                            if (wsum > 0.0) {
                                for (const TurnEnd& e : fin) mean += e.eval * (double)e.visits / wsum;
                                double var = 0.0;
                                for (const TurnEnd& e : fin) {
                                    const double d = e.eval - mean;
                                    var += d * d * (double)e.visits / wsum;
                                }
                                const double spread = std::sqrt(std::max(0.0, var));
                                want = adaptBaseSec * (spread / 0.0360);
                                g_gapSum += spread; g_wantSum += want / std::max(1e-9, adaptBaseSec); ++g_gapN;
                                // Whether adaptivity can win at all hinges on how
                                // widely importance varies: time should go as the
                                // importance, and that only pays if the spread of
                                // importances is a matter of several times over.
                                if (g_spreadN < 200000) g_spreadSamples[g_spreadN++] = spread;
                            }
                        }
                        if (adaptMode == 8) {
                            // How much the search is still learning: for every
                            // outcome already known last time, how far the
                            // verdict of its new visits departs from the verdict
                            // of the old ones, weighted by how many new visits
                            // there were. Outcomes seen for the first time
                            // contribute nothing - there is nothing to compare
                            // them against. While this stays large the picture
                            // is still moving and the search should go on.
                            double churn = 0.0, newTotal = 0.0;
                            for (const TurnEnd& now : fin) {
                                for (const TurnEnd& was : prevEnds) {
                                    if (was.key != now.key) continue;
                                    const double dv = (double)now.visits - (double)was.visits;
                                    if (dv <= 0.0 || was.visits == 0) break;
                                    const double avgNew = (now.sum - was.sum) / dv;
                                    const double avgOld = was.sum / (double)was.visits;
                                    churn += dv * std::fabs(avgNew - avgOld);
                                    newTotal += dv;
                                    break;
                                }
                            }
                            prevEnds = fin;
                            if (newTotal > 0.0) {
                                const double moving = churn / newTotal;
                                want = adaptBaseSec * (moving / 0.030);
                                g_gapSum += moving;
                                g_wantSum += want / std::max(1e-9, adaptBaseSec);
                                ++g_gapN;
                                if (g_spreadN < 200000) g_spreadSamples[g_spreadN++] = moving;
                            }
                            else want = ceiling;
                        }
                        if (adaptMode == 4) {
                            // Visits only. The move played is the most visited
                            // one, so how firmly it leads is the natural measure
                            // of whether the decision can still change - and
                            // unlike an evaluation it cannot be noisy.
                            uint64_t topAlt = 1;
                            for (size_t i = 0; i < fin.size(); ++i)
                                if (i != mineIdx) topAlt = std::max(topAlt, fin[i].visits);
                            const double lead = (double)mineVisits / (double)topAlt;
                            // Constant fixed from measurement rather than from a
                            // feedback loop: the loop kept trading one bias for
                            // another and never settled, and a stable budget
                            // matters more here than a self-tuning one.
                            want = adaptBaseSec * (1.30 / std::max(lead, 0.5));
                        }
                        if (adaptMode == 3) {
                            // Cost of being wrong times the chance of still
                            // being wrong: a leader that has pulled away on
                            // visits will not be overtaken, however close the
                            // evaluations are.
                            const double lead = (double)mineVisits / (double)altVisits;
                            // Symmetric on purpose: a leader that has not pulled
                            // away yet buys more time, one that has gives it
                            // back. Capping this at 1 would only ever shave time
                            // off and the sides would not be spending equally.
                            want *= 1.6 / std::max(0.5, lead);
                        }
                        // Whatever the metric, unspent time has to come back:
                        // a rule that only ever shaves seconds off is not a
                        // redistribution, it is a smaller budget, and it would
                        // lose the comparison for the wrong reason.
                        // Closed loop on the actual spend: the caller measures
                        // how much of the allowance is really being used and
                        // passes back the correction. Hand-tuned constants kept
                        // drifting below the budget, which turns redistribution
                        // into a quiet time cut.
                        // The floor is what actually sets the average: stops only
                        // ever happen on small targets, the large ones never get
                        // reached, so too low a floor quietly shrinks the budget.
                        // The controller has to move the floor as well: stops
                        // land on it, so leaving it fixed pins the average below
                        // the allowance no matter what the target says.
                        const double scale = adaptCredit > 0.0 ? adaptCredit : 1.0;
                        want *= scale;
                        want = std::min(std::max(want, adaptBaseSec * 0.15 * scale), ceiling);
                        g_gapSum += gap; g_wantSum += want / std::max(1e-9, adaptBaseSec); ++g_gapN;
                        if (elapsed >= want) {
                            ++g_adaptStopSettled;
                            g_probeSec += std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - tProbe).count();
                            break;
                        }
                    }
                    // Walking the tree is not free, and if it eats into the
                    // search it would poison the very comparison it serves.
                    g_probeSec += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - tProbe).count();
                }
            }
        }

        if (adaptMode == 1 && ++adaptPoll % 12 == 0) {
            const double elapsed =
                std::chrono::duration<double>(now - tStart).count();
            if (elapsed >= adaptBaseSec * 0.2 &&
                rootNode->expanded.load(std::memory_order_acquire) == 1 &&
                rootNode->edgeCount) {
                // Cheap test first: who leads and by how much. Reading the
                // edge counters costs nothing, while walking every root move's
                // principal variation would steal real work from the workers.
                TTEdge* e0 = T.edgePtr(rootNode->edgeBegin);
                uint32_t v1 = 0, v2 = 0;
                int leader = 0;
                for (int i = 0; i < (int)rootNode->edgeCount; ++i) {
                    uint32_t v = e0[i].visits.load(std::memory_order_relaxed);
                    if (v > v1) { v2 = v1; v1 = v; leader = e0[i].move; }
                    else if (v > v2) v2 = v;
                }
                if (leader == adaptLeader) ++adaptStable;
                else { adaptLeader = leader; adaptStable = 0; }
                // Settled: the same move has led for a while and the runner-up
                // cannot catch it in the time that is left.
                if (adaptStable >= 2 && v1 > v2 * 3 / 2) { ++g_adaptStopSettled; break; }

                // Occasionally ask the expensive question too: does the leading
                // move cut any reachable position off? If it does not, playing
                // it postpones the decision at no cost and thinking on is waste.
                if (adaptPoll % 75 == 0) {
                    std::vector<moveState> snap;
                    collectRootMoves(T, rootPos, qIgnored, snap);
                    if (!snap.empty() && snap[0].visits) {
                        for (auto& ms : snap) {
                            Position p = rootPos;
                            makeMove(p, mask, ms.move);
                            extractBestPVUntilChance(T, p, mask, ms.pv, ms.pvKey);
                            ms.pv.insert(ms.pv.begin(), ms.move);
                        }
                        computeDifForRootMoves(0, snap, T, rootPos, mask);
                        if (!snap.empty() && snap[0].dif >= 99.0) { ++g_adaptStopFree; break; }
                    }
                }
            }
        }

        if (write == 2) {
            if (now >= tNextWrite) {
                emitSearchSnapshot();
                tNextWrite += std::chrono::seconds(1);
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    stop.store(true, std::memory_order_relaxed);

    for (auto& th : pool) th.join();
    poolGuard.release();

    if (extServer) extServer->waitIdle(); // drain in-flight results into the tree, keep server alive
    else {
        localServer.stopAndDrain();
        nnServerGuard.release();
    }

    stopGuard.release();

    float qRoot = nodeQ(*rootNode);
    outEvalWhite = (rootPos.side == 0) ? qRoot : (1.0f - qRoot);
    {
        const uint64_t simsNow = simOK.load(std::memory_order_relaxed);
        const uint64_t depthNow = depthSum.load(std::memory_order_relaxed);
        outAvgDepth = simsNow ? (float)((double)depthNow / (double)simsNow) : 0.0f;
    }

    outRootMoves.clear();
    uint8_t ex = rootNode->expanded.load(std::memory_order_acquire);
    if (ex == 1 && rootNode->edgeCount) {
        TTEdge* e0 = T.edgePtr(rootNode->edgeBegin);
        outRootMoves.reserve(rootNode->edgeCount);

        for (int i = 0; i < (int)rootNode->edgeCount; ++i) {
            const TTEdge& e = e0[i];
            uint32_t v = e.visits.load(std::memory_order_relaxed);
            float p = e.prior();
            float ev = -1.0f;
            if (v) ev = clamp01(e.sum() / (float)v);

            outRootMoves.push_back(moveState{ e.move, ev, v, p, 0ull, {} });
        }

        std::sort(outRootMoves.begin(), outRootMoves.end(),
            [](const moveState& a, const moveState& b) {
                if (a.visits != b.visits) return a.visits > b.visits;
                return a.eval > b.eval;
            });

        if (rootPos.side == 1) {
            for (auto& ms : outRootMoves) {
                if (ms.eval >= 0.0f) ms.eval = 1.0f - ms.eval;
            }
        }
        for (auto& ms : outRootMoves) {
            Position p = rootPos;
            makeMove(p, mask, ms.move);
            extractBestPVUntilChance(T, p, mask, ms.pv, ms.pvKey);
            ms.pv.insert(ms.pv.begin(), ms.move);
        }
    }

    if (winMove) {
        // Hand the winning move back as the one the search settled on.
        for (auto& ms : outRootMoves) if (ms.move == winMove) ms.visits = UINT32_MAX;
        std::stable_sort(outRootMoves.begin(), outRootMoves.end(),
            [](const moveState& a, const moveState& b) { return a.visits > b.visits; });
    }

    if (write == 0) {
        // Playing mode wants the line the search actually settled on: the most
        // visited move at every step. The dif ordering is a reporting device
        // and has no business steering the move that gets played.
        uint64_t pvKey = 0;
        outPVBeforeRoll.clear();
        extractBestPVUntilChance(T, rootPos, mask, outPVBeforeRoll, pvKey);
    }
    else {
        computeDifForRootMoves(write, outRootMoves, T, rootPos, mask);
        extractDifPVUntilChance(write, T, rootPos, mask, outRootMoves, outPVBeforeRoll);
    }
    if (winMove) {
        outPVBeforeRoll.clear();
        outPVBeforeRoll.push_back(winMove);
    }

    (void)simOK; (void)simFail; (void)nnExp;
    if (forceExit) return;
}

// ===================== TRAINING PATCH BEGIN (FINAL) =====================
// (continuation will be in message 2/2)
// ===================== TRAINING PATCH BEGIN (FINAL) =====================
// INSERT THIS INSTEAD OF YOUR CURRENT `static void init()` AND `int main()`
// (i.e., delete/replace everything from `static void init()` to the end of file).



// ========================= Torch BN+SE Net (matches TRT) =========================

// ========================= Torch BN + Affine-SE Net (10x128) =========================


struct SEAffineImpl final : torch::nn::Module {
    int C = 0;
    int seC = 0;

    torch::nn::AdaptiveAvgPool2d pool{ nullptr };
    torch::nn::Conv2d fc1{ nullptr }, fc2{ nullptr };

    SEAffineImpl(int channels, int seChannels) : C(channels), seC(seChannels) {
        pool = register_module("pool",
            torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({ 1,1 })));

        fc1 = register_module("fc1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(C, seC, 1).padding(0).bias(true)));

        // outputs 2*C => split into W and B
        fc2 = register_module("fc2",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(seC, 2 * C, 1).padding(0).bias(true)));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto s = pool->forward(x);            // [B,C,1,1]
        s = torch::relu(fc1->forward(s));     // [B,seC,1,1]
        s = fc2->forward(s);                  // [B,2C,1,1]

        auto W = s.slice(1, 0, C);            // [B,C,1,1]
        auto B = s.slice(1, C, 2 * C);          // [B,C,1,1]

        auto Z = torch::sigmoid(W);
        return Z * x + B;
    }
};
TORCH_MODULE(SEAffine);

struct ResBlockImpl final : torch::nn::Module {
    int C = 0;

    torch::nn::Conv2d conv1{ nullptr }, conv2{ nullptr };
    torch::nn::BatchNorm2d bn1{ nullptr }, bn2{ nullptr };
    SEAffine se{ nullptr };

    explicit ResBlockImpl(int channels, int seChannels = SE_CHANNELS) : C(channels) {
        conv1 = register_module("conv1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(C, C, 3).padding(1).bias(false)));
        bn1 = register_module("bn1",
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(C).eps(AI_BN_EPS)));

        conv2 = register_module("conv2",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(C, C, 3).padding(1).bias(false)));
        bn2 = register_module("bn2",
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(C).eps(AI_BN_EPS)));

        se = register_module("se", SEAffine(C, seChannels));
    }

    torch::Tensor forward(torch::Tensor x) {
        auto skip = x;
        x = torch::relu(bn1->forward(conv1->forward(x)));
        x = bn2->forward(conv2->forward(x));
        x = se->forward(x);
        x = torch::relu(x + skip);
        return x;
    }
};
TORCH_MODULE(ResBlock);

struct LegacyNetImpl final : torch::nn::Module {
    torch::nn::Conv2d stem{ nullptr };
    torch::nn::BatchNorm2d stemBn{ nullptr };
    torch::nn::ModuleList blocks;

    torch::nn::Conv2d polConv1{ nullptr };
    torch::nn::BatchNorm2d polBn1{ nullptr };
    torch::nn::Conv2d polConv2{ nullptr };

    torch::nn::Conv2d valConv1{ nullptr };
    torch::nn::BatchNorm2d valBn1{ nullptr };
    torch::nn::Linear valFC1{ nullptr };
    torch::nn::Linear valFC2{ nullptr };

    LegacyNetImpl() {
        stem = register_module("stem",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(LEGACY_NN_SQ_PLANES, NET_CHANNELS, 3).padding(1).bias(false)));
        stemBn = register_module("stemBn",
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(NET_CHANNELS).eps(AI_BN_EPS)));

        blocks = register_module("blocks", torch::nn::ModuleList());
        for (int i = 0; i < NET_BLOCKS; ++i) blocks->push_back(ResBlock(NET_CHANNELS));

        polConv1 = register_module("polConv1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(NET_CHANNELS, HEAD_POLICY_C, 1).padding(0).bias(false)));
        polBn1 = register_module("polBn1",
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(HEAD_POLICY_C).eps(AI_BN_EPS)));
        polConv2 = register_module("polConv2",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(HEAD_POLICY_C, POLICY_P, 1).padding(0).bias(true)));

        valConv1 = register_module("valConv1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(NET_CHANNELS, HEAD_VALUE_C, 1).padding(0).bias(false)));
        valBn1 = register_module("valBn1",
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(HEAD_VALUE_C).eps(AI_BN_EPS)));

        valFC1 = register_module("valFC1",
            torch::nn::Linear(torch::nn::LinearOptions(HEAD_VALUE_C * 64, HEAD_VALUE_FC).bias(true)));
        valFC2 = register_module("valFC2",
            torch::nn::Linear(torch::nn::LinearOptions(HEAD_VALUE_FC, 1).bias(true)));
    }
};
TORCH_MODULE(LegacyNet);

struct NetImpl final : torch::nn::Module {
    torch::nn::Conv2d stem{ nullptr };
    torch::nn::BatchNorm2d stemBn{ nullptr };
    torch::nn::ModuleList blocks;

    // policy: 1x1 -> BN -> ReLU -> 1x1 logits
    torch::nn::Conv2d polConv1{ nullptr };
    torch::nn::BatchNorm2d polBn1{ nullptr };
    torch::nn::Conv2d polConv2{ nullptr };

    // value: 1x1 -> BN -> ReLU -> flatten -> FC -> ReLU -> FC -> Sigmoid
    torch::nn::Conv2d valConv1{ nullptr };
    torch::nn::BatchNorm2d valBn1{ nullptr };
    torch::nn::Linear valFC1{ nullptr };
    torch::nn::Linear valFC2{ nullptr };

    NetImpl() {
        stem = register_module("stem",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(NN_SQ_PLANES, NET_CHANNELS, 3).padding(1).bias(false)));
        stemBn = register_module("stemBn",
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(NET_CHANNELS).eps(AI_BN_EPS)));

        blocks = register_module("blocks", torch::nn::ModuleList());
        for (int i = 0; i < NET_BLOCKS; ++i) blocks->push_back(ResBlock(NET_CHANNELS));

        polConv1 = register_module("polConv1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(NET_CHANNELS, HEAD_POLICY_C, 1).padding(0).bias(false)));
        polBn1 = register_module("polBn1",
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(HEAD_POLICY_C).eps(AI_BN_EPS)));
        polConv2 = register_module("polConv2",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(HEAD_POLICY_C, POLICY_P, 1).padding(0).bias(true)));

        valConv1 = register_module("valConv1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(NET_CHANNELS, HEAD_VALUE_C, 1).padding(0).bias(false)));
        valBn1 = register_module("valBn1",
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(HEAD_VALUE_C).eps(AI_BN_EPS)));

        valFC1 = register_module("valFC1",
            torch::nn::Linear(torch::nn::LinearOptions(HEAD_VALUE_C * 64, HEAD_VALUE_FC).bias(true)));
        valFC2 = register_module("valFC2",
            torch::nn::Linear(torch::nn::LinearOptions(HEAD_VALUE_FC, 1).bias(true)));
    }

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
        x = torch::relu(stemBn->forward(stem->forward(x)));
        for (int i = 0; i < NET_BLOCKS; ++i)
            x = blocks[i]->as<ResBlock>()->forward(x);

        auto pol = polConv2->forward(torch::relu(polBn1->forward(polConv1->forward(x))));

        auto v = torch::relu(valBn1->forward(valConv1->forward(x)));
        v = v.contiguous().view({ v.size(0), HEAD_VALUE_C * 64 });
        v = torch::relu(valFC1->forward(v));

        // Get raw logits
        v = valFC2->forward(v);


        if (!is_training()) {
            v = torch::sigmoid(v);
        }

        return { pol, v };
    }
};
TORCH_MODULE(Net);

// ============================================================
// widen192: expand a 10x128(se16) checkpoint into this build's
// 10x192(se24) Net, preserving the source function exactly:
// weights consuming NEW input channels are zeroed for the rows
// that produce ORIGINAL outputs, so both heads compute the same
// function as the 128 net until training turns new capacity on.
// ============================================================

// Donor module: exact 10x128/se16/p32/v32 architecture (same submodule names as Net).
struct Net128Impl final : torch::nn::Module {
    static constexpr int C128 = 128;
    static constexpr int SE128 = 16;

    torch::nn::Conv2d stem{ nullptr };
    torch::nn::BatchNorm2d stemBn{ nullptr };
    torch::nn::ModuleList blocks;

    torch::nn::Conv2d polConv1{ nullptr };
    torch::nn::BatchNorm2d polBn1{ nullptr };
    torch::nn::Conv2d polConv2{ nullptr };

    torch::nn::Conv2d valConv1{ nullptr };
    torch::nn::BatchNorm2d valBn1{ nullptr };
    torch::nn::Linear valFC1{ nullptr };
    torch::nn::Linear valFC2{ nullptr };

    Net128Impl() {
        stem = register_module("stem",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(NN_SQ_PLANES, C128, 3).padding(1).bias(false)));
        stemBn = register_module("stemBn",
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(C128).eps(AI_BN_EPS)));

        blocks = register_module("blocks", torch::nn::ModuleList());
        for (int i = 0; i < NET_BLOCKS; ++i) blocks->push_back(ResBlock(C128, SE128));

        polConv1 = register_module("polConv1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(C128, HEAD_POLICY_C, 1).padding(0).bias(false)));
        polBn1 = register_module("polBn1",
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(HEAD_POLICY_C).eps(AI_BN_EPS)));
        polConv2 = register_module("polConv2",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(HEAD_POLICY_C, POLICY_P, 1).padding(0).bias(true)));

        valConv1 = register_module("valConv1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(C128, HEAD_VALUE_C, 1).padding(0).bias(false)));
        valBn1 = register_module("valBn1",
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(HEAD_VALUE_C).eps(AI_BN_EPS)));

        valFC1 = register_module("valFC1",
            torch::nn::Linear(torch::nn::LinearOptions(HEAD_VALUE_C * 64, HEAD_VALUE_FC).bias(true)));
        valFC2 = register_module("valFC2",
            torch::nn::Linear(torch::nn::LinearOptions(HEAD_VALUE_FC, 1).bias(true)));
    }
};
TORCH_MODULE(Net128);

struct WidenCopyStats {
    int exact = 0;
    int prefix = 0;
    int seSplit = 0;
    int zeroedCross = 0;
};

static bool widenEndsWith(const std::string& s, const char* suffix) {
    const size_t n = std::strlen(suffix);
    return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

static torch::Tensor widenTensorForCopy(const torch::Tensor& src, const torch::Tensor& dst) {
    return src.detach().to(dst.device(), dst.scalar_type(), false, false).contiguous();
}

static bool widenSameShape(const torch::Tensor& a, const torch::Tensor& b) {
    if (a.dim() != b.dim()) return false;
    for (int64_t d = 0; d < a.dim(); ++d)
        if (a.size(d) != b.size(d)) return false;
    return true;
}

static bool widenFitsPrefix(const torch::Tensor& src, const torch::Tensor& dst) {
    if (src.dim() != dst.dim()) return false;
    for (int64_t d = 0; d < src.dim(); ++d)
        if (src.size(d) > dst.size(d)) return false;
    return true;
}

static torch::Tensor widenPrefixView(const torch::Tensor& dst, const torch::Tensor& src) {
    torch::Tensor view = dst;
    for (int64_t d = 0; d < src.dim(); ++d)
        view = view.slice(d, 0, src.size(d));
    return view;
}

// For weights with an input-channel dim (conv [O,I,k,k], linear [O,I]):
// zero dst[0:srcO, srcI:dstI] so original outputs ignore new channels.
static void widenZeroCrossTerms(torch::Tensor& dst,
    const torch::Tensor& src,
    WidenCopyStats& stats) {
    if (src.dim() < 2) return;
    const int64_t srcO = src.size(0), srcI = src.size(1);
    const int64_t dstI = dst.size(1);
    if (srcI >= dstI) return;
    torch::Tensor z = dst.slice(0, 0, srcO).slice(1, srcI, dstI);
    z.zero_();
    ++stats.zeroedCross;
}

// SE fc2 emits 2*C (W then B halves): copy each half's prefix and zero its cross-terms.
static bool widenCopySplitSEFc2(const std::string& name,
    torch::Tensor dst,
    const torch::Tensor& srcIn,
    WidenCopyStats& stats) {
    const bool isWeight = widenEndsWith(name, ".se.fc2.weight");
    const bool isBias = widenEndsWith(name, ".se.fc2.bias");
    if (!isWeight && !isBias) return false;

    if (srcIn.dim() != dst.dim() || srcIn.dim() < 1)
        throw std::runtime_error("SE fc2 tensor rank mismatch: " + name);
    if ((srcIn.size(0) % 2) != 0 || (dst.size(0) % 2) != 0)
        throw std::runtime_error("SE fc2 channel mismatch: " + name);

    const int64_t srcC = srcIn.size(0) / 2;
    const int64_t dstC = dst.size(0) / 2;
    if (srcC > dstC)
        throw std::runtime_error("SE fc2 cannot shrink: " + name);

    const torch::Tensor s = widenTensorForCopy(srcIn, dst);

    torch::Tensor srcW = s.slice(0, 0, srcC);
    torch::Tensor srcB = s.slice(0, srcC, srcC * 2);
    torch::Tensor dstW = dst.slice(0, 0, srcC);
    torch::Tensor dstB = dst.slice(0, dstC, dstC + srcC);

    for (int64_t d = 1; d < s.dim(); ++d) {
        dstW = dstW.slice(d, 0, s.size(d));
        dstB = dstB.slice(d, 0, s.size(d));
    }
    dstW.copy_(srcW);
    dstB.copy_(srcB);

    if (isWeight && s.dim() >= 2 && s.size(1) < dst.size(1)) {
        // zero new-seC inputs for the copied W and B rows
        torch::Tensor zw = dst.slice(0, 0, srcC).slice(1, s.size(1), dst.size(1));
        torch::Tensor zb = dst.slice(0, dstC, dstC + srcC).slice(1, s.size(1), dst.size(1));
        zw.zero_();
        zb.zero_();
        ++stats.zeroedCross;
    }

    ++stats.seSplit;
    return true;
}

static void widenCopyTensorByName(const std::string& name,
    torch::Tensor dst,
    const torch::Tensor& src,
    WidenCopyStats& stats) {
    torch::NoGradGuard ng;

    if (!widenSameShape(src, dst) && widenCopySplitSEFc2(name, dst, src, stats)) return;

    const torch::Tensor s = widenTensorForCopy(src, dst);
    if (widenSameShape(s, dst)) {
        dst.copy_(s);
        ++stats.exact;
        return;
    }

    if (!widenFitsPrefix(s, dst)) {
        std::ostringstream oss;
        oss << "widen192: cannot copy '" << name << "'";
        throw std::runtime_error(oss.str());
    }

    widenPrefixView(dst, s).copy_(s);
    widenZeroCrossTerms(dst, s, stats);
    ++stats.prefix;
}

static WidenCopyStats widenNet128To192(Net128& src, Net& dst) {
    WidenCopyStats stats;
    torch::NoGradGuard ng;

    auto srcParams = src->named_parameters(true);
    auto dstParams = dst->named_parameters(true);
    for (const auto& kv : srcParams) {
        auto* d = dstParams.find(kv.key());
        if (!d) throw std::runtime_error("widen192 missing parameter: " + kv.key());
        widenCopyTensorByName(kv.key(), *d, kv.value(), stats);
    }

    auto srcBufs = src->named_buffers(true);
    auto dstBufs = dst->named_buffers(true);
    for (const auto& kv : srcBufs) {
        auto* d = dstBufs.find(kv.key());
        if (!d) throw std::runtime_error("widen192 missing buffer: " + kv.key());
        widenCopyTensorByName(kv.key(), *d, kv.value(), stats);
    }

    dst->eval();
    return stats;
}

static bool createNet192FromFile(const std::string& srcFile, const std::string& dstFile) {
    try {
        Net128 src;
        torch::load(src, srcFile);
        src->to(torch::kCPU);
        src->eval();

        torch::manual_seed(192);
        Net dst;
        dst->to(torch::kCPU);

        const WidenCopyStats stats = widenNet128To192(src, dst);
        torch::save(dst, dstFile);

        std::cout << "widen192: " << srcFile << " -> " << dstFile
            << " (exact=" << stats.exact
            << ", prefix=" << stats.prefix
            << ", seSplit=" << stats.seSplit
            << ", zeroedCross=" << stats.zeroedCross << ")" << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "widen192 failed for " << srcFile << ": " << e.what() << std::endl;
        return false;
    }
}

// ------------------------------------------------------------
// ReplayBuffer: X + SPARSE policy target + z
// ------------------------------------------------------------

struct TrainSample {
    Position pos;

    // sparse policy target:
    // idx = CHW index in [0..POLICY_SIZE-1], i.e. pl*64 + sq
    uint16_t nPi = 0;
    std::array<uint16_t, AI_MAX_MOVES> piIdx{};
    std::array<uint16_t, AI_MAX_MOVES> piProbQ{}; // quantized probs in [0..65535]
    float q = 0.5f;
    float z = 0.5f; // [0..1] from side-to-move perspective
};

static AI_FORCEINLINE void decodeTrainSamplePolicyRow(
    const TrainSample& s,
    int64_t* idxRow,
    float* probRow) {
    const int n = std::min<int>((int)s.nPi, AI_MAX_MOVES);

    for (int j = 0; j < AI_MAX_MOVES; ++j) {
        idxRow[(size_t)j] = (int64_t)s.piIdx[(size_t)j];
        probRow[(size_t)j] = 0.0f;
    }

    double sum = 0.0;
    for (int j = 0; j < n; ++j) {
        float p = dequantizeProbU16(s.piProbQ[(size_t)j]);
        if (!(p >= 0.0f) || !std::isfinite(p)) p = 0.0f;
        probRow[(size_t)j] = p;
        sum += (double)p;
    }

    if (n <= 0) return;

    if (sum > 0.0) {
        const float inv = (float)(1.0 / sum);
        for (int j = 0; j < n; ++j) {
            probRow[(size_t)j] *= inv;
        }
    }
    else {
        const float inv = 1.0f / (float)n;
        for (int j = 0; j < n; ++j) {
            probRow[(size_t)j] = inv;
        }
    }
}

struct ReplayBuffer {
    std::vector<TrainSample> buf;
    size_t cap = 16384;
    size_t head = 0;
    size_t size = 0;

    // Degree of data "freshness" (Prioritized Replay Lite).
    // 1.0  = fully uniform sampling (as before).
    // 0.75 = slight priority to fresh games (a sweet spot for AlphaZero).
    // 0.5  = strong skew toward just-played games.
    double recent_bias = 0.85;

    std::mutex m;

    explicit ReplayBuffer(size_t capacity) : cap(capacity) {
        buf.resize(cap);
    }

    void push(const TrainSample& s) {
        std::lock_guard<std::mutex> lk(m);
        buf[head] = s;
        head = (head + 1) % cap;
        if (size < cap) ++size;
    }
    void pushMany(const std::vector<TrainSample>& v) {
        if (v.empty() || cap == 0) return;

        std::lock_guard<std::mutex> lk(m);

        size_t n = v.size();

        // If incoming batch is bigger than the whole buffer,
        // keep only the newest `cap` samples.
        if (n >= cap) {
            std::copy(v.end() - cap, v.end(), buf.begin());
            head = 0;
            size = cap;
            return;
        }

        size_t space_at_end = cap - head;

        if (n <= space_at_end) {
            std::copy(v.begin(), v.end(), buf.begin() + head);
        }
        else {
            std::copy(v.begin(), v.begin() + space_at_end, buf.begin() + head);
            std::copy(v.begin() + space_at_end, v.end(), buf.begin());
        }

        head = (head + n) % cap;
        size = std::min(cap, size + n);
    }
    bool sampleBatch(std::vector<TrainSample>& out, int B, std::mt19937& rng) {
        out.resize((size_t)B);

        std::vector<double> biased((size_t)B);
        std::uniform_real_distribution<double> d(0.0, 1.0);
        for (int i = 0; i < B; ++i) {
            biased[(size_t)i] = std::pow(d(rng), recent_bias);
        }

        std::lock_guard<std::mutex> lk(m);
        if (size < (size_t)B) return false;

        const size_t snapSize = size;
        const size_t start = (head + cap - snapSize) % cap;

        for (int i = 0; i < B; ++i) {
            size_t li = (size_t)(biased[(size_t)i] * (double)snapSize);
            if (li >= snapSize) li = snapSize - 1;
            out[(size_t)i] = buf[(start + li) % cap];
        }
        return true;
    }

    size_t currentSize() {
        std::lock_guard<std::mutex> lk(m);
        return size;
    }

    // ---- persistence for fast training restart ----
    static constexpr uint64_t REPLAY_MAGIC = 0xD1CE0BAFull;

    bool saveToFile(const std::string& path) {
        static_assert(std::is_trivially_copyable<TrainSample>::value,
            "TrainSample must be trivially copyable for raw dump");

        std::vector<TrainSample> snap;
        {
            std::lock_guard<std::mutex> lk(m);
            snap.resize(size);
            const size_t start = (head + cap - size) % cap;
            for (size_t i = 0; i < size; ++i) snap[i] = buf[(start + i) % cap];
        }

        const std::string tmp = path + ".tmp";
        {
            std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
            if (!f) return false;
            const uint64_t magic = REPLAY_MAGIC;
            const uint32_t version = 1;
            const uint32_t ss = (uint32_t)sizeof(TrainSample);
            const uint64_t count = (uint64_t)snap.size();
            f.write((const char*)&magic, 8);
            f.write((const char*)&version, 4);
            f.write((const char*)&ss, 4);
            f.write((const char*)&count, 8);
            if (count) f.write((const char*)snap.data(),
                (std::streamsize)(count * sizeof(TrainSample)));
            if (!f.good()) return false;
        }
#if defined(_WIN32)
        if (!MoveFileExA(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING)) return false;
#else
        if (std::rename(tmp.c_str(), path.c_str()) != 0) return false;
#endif
        return true;
    }

    size_t loadFromFile(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return 0;

        uint64_t magic = 0; uint32_t version = 0, ss = 0; uint64_t count = 0;
        f.read((char*)&magic, 8);
        f.read((char*)&version, 4);
        f.read((char*)&ss, 4);
        f.read((char*)&count, 8);
        if (!f.good() || magic != REPLAY_MAGIC || version != 1 ||
            ss != (uint32_t)sizeof(TrainSample)) return 0;

        std::lock_guard<std::mutex> lk(m);
        const uint64_t keep = std::min<uint64_t>(count, (uint64_t)cap);
        if (count > keep) {
            f.seekg((std::streamoff)((count - keep) * sizeof(TrainSample)), std::ios::cur);
        }
        f.read((char*)buf.data(), (std::streamsize)(keep * sizeof(TrainSample)));
        if (!f.good()) { size = 0; head = 0; return 0; }
        size = (size_t)keep;
        head = (size_t)(keep % cap);
        return size;
    }
};

// ------------------------------------------------------------
// TRT refit from libtorch model + Context rebuild + CUDA Graph
// ------------------------------------------------------------

static std::mutex g_trtMutex;     // protects TRT enqueue/refit/serialize
static std::mutex g_modelMutex;   // protects model weight read/write and optimizer step
static TrtRunner g_trt_old;
static bool g_trtOldReady = false;
static std::mutex g_trtOldMutex;

struct BackendBinding {
    TrtRunner& trt;
    std::mutex& mtx;
};
// Always lock BOTH in the same order, deadlock-free (C++17)
static AI_FORCEINLINE std::scoped_lock<std::mutex, std::mutex> lockModelTrt() {
    return std::scoped_lock<std::mutex, std::mutex>(g_modelMutex, g_trtMutex);
}
static AI_FORCEINLINE nvinfer1::Weights trtWeightsFromVec(const std::vector<float>& v) {
    nvinfer1::Weights w{};
    w.type = nvinfer1::DataType::kFLOAT;
    w.values = v.data();
    w.count = (int64_t)v.size();
    return w;
}

static std::vector<float> tensorToHostVecF32(const torch::Tensor& tIn) {
    torch::Tensor t = tIn.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    std::vector<float> v((size_t)t.numel());
    std::memcpy(v.data(), t.data_ptr<float>(), v.size() * sizeof(float));
    return v;
}

// Pretty-print missing refit weights (IMPORTANT: otherwise refit may be silently partial).
static void trtDumpMissingRefitWeights(nvinfer1::IRefitter& ref) {
    using namespace nvinfer1;

    const int32_t nMiss = ref.getMissing(0, nullptr, nullptr);
    if (nMiss <= 0) return;

    std::vector<const char*> layerNames((size_t)nMiss);
    std::vector<WeightsRole> roles((size_t)nMiss);

    const int32_t n2 = ref.getMissing(nMiss, layerNames.data(), roles.data());
    std::cerr << "[TRT][refit] Missing weights: " << n2 << "\n";
    for (int32_t i = 0; i < n2; ++i) {
        const char* ln = layerNames[(size_t)i] ? layerNames[(size_t)i] : "<null>";
        const int rr = (int)roles[(size_t)i];
        std::cerr << "  - layer='" << ln << "' role=" << rr << "\n";
    }

    const int32_t nMW = ref.getMissingWeights(0, nullptr);
    if (nMW > 0) {
        std::vector<const char*> wnames((size_t)nMW);
        const int32_t n3 = ref.getMissingWeights(nMW, wnames.data());
        std::cerr << "[TRT][refit] MissingWeights(names): " << n3 << "\n";
        for (int32_t i = 0; i < n3; ++i) {
            const char* wn = wnames[(size_t)i] ? wnames[(size_t)i] : "<unnamed>";
            std::cerr << "  - weightName='" << wn << "'\n";
        }
    }
}

static bool trtRecreateContextAndRebindAndGraph(TrtRunner& trt) {
    using namespace nvinfer1;

    if (!trt.engine || !trt.stream) return false;

    CUDA_CHECK(cudaStreamSynchronize(trt.stream));

    if (trt.graphExec) { cudaGraphExecDestroy(trt.graphExec); trt.graphExec = nullptr; }
    if (trt.graph) { cudaGraphDestroy(trt.graph);         trt.graph = nullptr; }
    trt.graphReady = false;

    IExecutionContext* newCtx = trt.engine->createExecutionContext();
    if (!newCtx) {
        std::cerr << "TensorRT: createExecutionContext() failed after refit.\n";
        return false;
    }

    if (!newCtx->setTensorAddress("policy", trt.dPolicy)) { delete newCtx; return false; }
    if (!newCtx->setTensorAddress("value", trt.dValue)) { delete newCtx; return false; }
    if (!newCtx->setInputTensorAddress("input", trt.dInput)) { delete newCtx; return false; }

    if (!newCtx->setOptimizationProfileAsync(0, trt.stream)) { delete newCtx; return false; }
    if (trt.ctx) { delete trt.ctx; trt.ctx = nullptr; }
    trt.ctx = newCtx;
    trt.currentShapeB = -1;

    if (!trt.ensureShape(TRT_MAX_BATCH)) {
        std::cerr << "TensorRT: setInputShape failed on new ctx.\n";
        return false;
    }

    // IMPORTANT: re-attach aux streams for the NEW context BEFORE capture
    if (!trt.setupAuxStreams()) {
        std::cerr << "TensorRT: setupAuxStreams failed on new ctx; capture may fail.\n";
        // continue anyway
    }

    if (!trt.captureCudaGraphFixed256()) {
        std::cerr << "TensorRT: CUDA Graph re-capture failed; continue without graph.\n";
        trt.graphReady = false;
    }
    return true;
}

// =============================================================
// TensorRT refit from Torch model (BN-as-Scale + Affine-SE) for 10x128
// Names MUST match your TRT builder:
//   stem.conv, stem.bn
//   block{i}.conv1, block{i}.bn1, block{i}.conv2, block{i}.bn2
//   block{i}.se.fc1, block{i}.se.fc2
//   head.policy.conv1, head.policy.bn1, head.policy.conv2
//   head.value.conv1, head.value.bn1
//   head.value.fc1.w, head.value.fc1.b, head.value.fc2.w, head.value.fc2.b
//
// IMPORTANT:
// - BN in TRT is a Scale layer => refit via WeightsRole::kSCALE / kSHIFT
// - Conv(no-bias) layers: refit KERNEL only (do NOT set BIAS)
// - Keep all host vectors alive until refitCudaEngine() finishes.
// =============================================================

// RAII: temporarily switch model to eval() during refit and restore mode afterward.
struct ScopedModelEval {
    Net& model;
    bool wasTraining = false;

    explicit ScopedModelEval(Net& m) : model(m) {
        wasTraining = model->is_training();
        model->eval();
    }
    ~ScopedModelEval() {
        if (wasTraining) model->train();
        else model->eval();
    }
};

static bool trtRefitFromTorchModel(TrtRunner& trt, Net& model) {
    using namespace nvinfer1;

    if (!trt.engine || !trt.ctx) return false;

    // IMPORTANT: perform refit from eval() so BN running stats do not change.
    ScopedModelEval evalGuard(model);
    torch::NoGradGuard ng;

    // If the model is on CUDA — synchronize if needed (optional but safe).
    try {
        auto params = model->parameters(); // std::vector<at::Tensor>
        if (!params.empty()) {
            auto dev = params.front().device();
            if (dev.is_cuda()) torch::cuda::synchronize();
        }
    }
    catch (...) {}

    std::unique_ptr<IRefitter> ref(createInferRefitter(*trt.engine, g_trtLogger));
    if (!ref) return false;

    // Keep host vectors alive (TensorRT reads weights during refitCudaEngine()).
    // std::deque guarantees stable element addresses.
    std::deque<std::vector<float>> keep;

    auto pushKeep = [&](std::vector<float>&& v) -> nvinfer1::Weights {
        keep.emplace_back(std::move(v));
        return trtWeightsFromVec(keep.back());
        };

    auto setConvNoBias = [&](const char* name, const torch::nn::Conv2d& c2d) -> bool {
        if (!ref->setWeights(name, WeightsRole::kKERNEL,
            pushKeep(tensorToHostVecF32(c2d->weight)))) {
            std::cerr << "[TRT][refit] setWeights(KERNEL) failed: " << name << "\n";
            return false;
        }
        return true;
        };

    auto setConvWithBias = [&](const char* name, const torch::nn::Conv2d& c2d) -> bool {
        if (!ref->setWeights(name, WeightsRole::kKERNEL,
            pushKeep(tensorToHostVecF32(c2d->weight)))) {
            std::cerr << "[TRT][refit] setWeights(KERNEL) failed: " << name << "\n";
            return false;
        }
        if (c2d->bias.defined()) {
            if (!ref->setWeights(name, WeightsRole::kBIAS,
                pushKeep(tensorToHostVecF32(c2d->bias)))) {
                std::cerr << "[TRT][refit] setWeights(BIAS) failed: " << name << "\n";
                return false;
            }
        }
        else {
            std::cerr << "[TRT][refit] Expected bias, but torch conv has no bias: " << name << "\n";
            return false;
        }
        return true;
        };

    // Torch BN => TRT Scale(SCALE/SHIFT)
    // scale = gamma / sqrt(var + eps)
    // shift = beta - mean * scale
    auto setBNScaleShift = [&](const char* name, const torch::nn::BatchNorm2d& bn) -> bool {
        torch::Tensor gamma = bn->weight.detach();
        torch::Tensor beta = bn->bias.detach();
        torch::Tensor mean = bn->running_mean.detach();
        torch::Tensor var = bn->running_var.detach();

        gamma = gamma.to(torch::kCPU).to(torch::kFloat32).contiguous();
        beta = beta.to(torch::kCPU).to(torch::kFloat32).contiguous();
        mean = mean.to(torch::kCPU).to(torch::kFloat32).contiguous();
        var = var.to(torch::kCPU).to(torch::kFloat32).contiguous();

        const int64_t C = gamma.numel();
        if (beta.numel() != C || mean.numel() != C || var.numel() != C) {
            std::cerr << "[TRT][refit] BN tensor size mismatch for: " << name << "\n";
            return false;
        }

        const float* g = gamma.data_ptr<float>();
        const float* b = beta.data_ptr<float>();
        const float* m = mean.data_ptr<float>();
        const float* v = var.data_ptr<float>();

        std::vector<float> scale((size_t)C);
        std::vector<float> shift((size_t)C);

        for (int64_t i = 0; i < C; ++i) {
            float s = g[i] / std::sqrt(v[i] + static_cast<float>(AI_BN_EPS));
            scale[(size_t)i] = s;
            shift[(size_t)i] = b[i] - m[i] * s;
        }

        if (!ref->setWeights(name, WeightsRole::kSCALE, pushKeep(std::move(scale)))) {
            std::cerr << "[TRT][refit] setWeights(SCALE) failed: " << name << "\n";
            return false;
        }
        if (!ref->setWeights(name, WeightsRole::kSHIFT, pushKeep(std::move(shift)))) {
            std::cerr << "[TRT][refit] setWeights(SHIFT) failed: " << name << "\n";
            return false;
        }
        return true;
        };

    // TRT FC weights/bias are Constant layers => role CONSTANT
    auto setConst = [&](const char* name, std::vector<float>&& v) -> bool {
        if (!ref->setWeights(name, WeightsRole::kCONSTANT, pushKeep(std::move(v)))) {
            std::cerr << "[TRT][refit] setWeights(CONSTANT) failed: " << name << "\n";
            return false;
        }
        return true;
        };

    // ------------------- Read weights from model -------------------

    // stem
    if (!setConvNoBias("stem.conv", model->stem)) return false;
    if (!setBNScaleShift("stem.bn", model->stemBn)) return false;

    // blocks
    for (int bi = 0; bi < NET_BLOCKS; ++bi) {
        auto blk = model->blocks[bi]->as<ResBlock>();

        {
            std::string n = "block" + std::to_string(bi) + ".conv1";
            if (!setConvNoBias(n.c_str(), blk->conv1)) return false;
        }
        {
            std::string n = "block" + std::to_string(bi) + ".bn1";
            if (!setBNScaleShift(n.c_str(), blk->bn1)) return false;
        }
        {
            std::string n = "block" + std::to_string(bi) + ".conv2";
            if (!setConvNoBias(n.c_str(), blk->conv2)) return false;
        }
        {
            std::string n = "block" + std::to_string(bi) + ".bn2";
            if (!setBNScaleShift(n.c_str(), blk->bn2)) return false;
        }

        // SE affine convs (bias=true)
        {
            std::string n1 = "block" + std::to_string(bi) + ".se.fc1";
            std::string n2 = "block" + std::to_string(bi) + ".se.fc2";
            if (!setConvWithBias(n1.c_str(), blk->se->fc1)) return false;
            if (!setConvWithBias(n2.c_str(), blk->se->fc2)) return false;
        }
    }

    // policy head
    if (!setConvNoBias("head.policy.conv1", model->polConv1)) return false;
    if (!setBNScaleShift("head.policy.bn1", model->polBn1)) return false;
    if (!setConvWithBias("head.policy.conv2", model->polConv2)) return false;

    // value head conv+bn
    if (!setConvNoBias("head.value.conv1", model->valConv1)) return false;
    if (!setBNScaleShift("head.value.bn1", model->valBn1)) return false;

    // value head FC constants
    {
        // Torch Linear weight: [out,in], TRT constant expects [in,out]
        auto w1 = model->valFC1->weight.detach(); // [HEAD_VALUE_FC, HEAD_VALUE_C*64]
        auto b1 = model->valFC1->bias.detach();   // [HEAD_VALUE_FC]
        auto w2 = model->valFC2->weight.detach(); // [1, HEAD_VALUE_FC]
        auto b2 = model->valFC2->bias.detach();   // [1]

        auto w1t = w1.transpose(0, 1).contiguous(); // [in,out]
        auto w2t = w2.transpose(0, 1).contiguous(); // [HEAD_VALUE_FC,1]

        if (!setConst("head.value.fc1.w", tensorToHostVecF32(w1t))) return false;
        if (!setConst("head.value.fc1.b", tensorToHostVecF32(b1.view({ 1, HEAD_VALUE_FC })))) return false;

        if (!setConst("head.value.fc2.w", tensorToHostVecF32(w2t))) return false;
        if (!setConst("head.value.fc2.b", tensorToHostVecF32(b2.view({ 1, 1 })))) return false;
    }

    if (trt.stream) CUDA_CHECK(cudaStreamSynchronize(trt.stream));

    // Verify no missing weights
    {
        const int32_t nMiss = ref->getMissing(0, nullptr, nullptr);
        if (nMiss > 0) {
            trtDumpMissingRefitWeights(*ref);
            std::cerr << "[TRT][refit] Aborting refit: missing weights present.\n";
            return false;
        }
    }

    if (!ref->refitCudaEngine()) {
        std::cerr << "[TRT][refit] refitCudaEngine() failed.\n";
        trtDumpMissingRefitWeights(*ref);
        return false;
    }

    // After refit: recreate context + rebind addresses + recapture CUDA graph
    if (!trtRecreateContextAndRebindAndGraph(trt)) {
        std::cerr << "[TRT][refit] Failed to recreate context/graph after refit.\n";
        return false;
    }

    return true;
}

static bool trtSavePlanToDisk(TrtRunner& trt, const std::string& planFile) {
    if (!trt.engine) return false;
    nvinfer1::IHostMemory* mem = trt.engine->serialize();
    if (!mem) return false;
    bool ok = writeFileAll(planFile, mem->data(), (size_t)mem->size());
    delete mem;
    return ok;
}





// ------------------------------------------------------------
// Inference server for training (CV instead of busy-wait), + g_trtMutex
// ------------------------------------------------------------
static std::atomic<int> g_inferInFlight{ 0 };
static std::atomic<uint64_t> g_inferBatchCount{ 0 };
static std::atomic<uint64_t> g_inferBatchSizeTotal{ 0 };
static std::atomic<uint64_t> g_inferBusyMicros{ 0 };

static AI_FORCEINLINE void recordInferBatchSize(int batchSize) {
    if (batchSize <= 0) return;
    g_inferBatchCount.fetch_add(1, std::memory_order_relaxed);
    g_inferBatchSizeTotal.fetch_add((uint64_t)batchSize, std::memory_order_relaxed);
}

static AI_FORCEINLINE double getAverageInferBatchSize() {
    const uint64_t cnt = g_inferBatchCount.load(std::memory_order_relaxed);
    if (cnt == 0) return 0.0;
    const uint64_t total = g_inferBatchSizeTotal.load(std::memory_order_relaxed);
    return (double)total / (double)cnt;
}

static AI_FORCEINLINE void recordInferBusyMicros(uint64_t us) {
    if (us == 0) return;
    g_inferBusyMicros.fetch_add(us, std::memory_order_relaxed);
}

struct InferInFlightGuard {
    InferInFlightGuard() { g_inferInFlight.fetch_add(1, std::memory_order_relaxed); }
    ~InferInFlightGuard() { g_inferInFlight.fetch_sub(1, std::memory_order_relaxed); }
};
struct ITrainInferenceServer {
    virtual ~ITrainInferenceServer() = default;

    virtual int size() const = 0;

    virtual bool submit(std::unique_ptr<PendingNN>&& job,
        const std::atomic<bool>* extCancel = nullptr,
        const std::atomic<bool>* extAbort = nullptr) = 0;

    virtual void waitIdle() = 0;
    virtual void requestStop() = 0;
    virtual void join() = 0;
};

struct UnifiedInferenceServerTrain : ITrainInferenceServer {
    BackendBinding backend;
    MCTSTable* defaultOwner = nullptr;   // nullptr => ownerT must be set per job
    int queueCap = 0;

    std::atomic<bool> stop{ false };
    std::atomic<int>  qSize{ 0 };

    std::mutex m;
    std::condition_variable cvNotEmpty;
    std::condition_variable cvNotFull;
    std::condition_variable cvIdle;

    std::deque<std::unique_ptr<PendingNN>> q;
    std::thread th;

    bool busyFlag = false;

    explicit UnifiedInferenceServerTrain(BackendBinding be,
        MCTSTable* fallbackOwner,
        int qCap)
        : backend(be), defaultOwner(fallbackOwner), queueCap(qCap) {
    }

    void start() {
        {
            std::lock_guard<std::mutex> lk(m);
            stop.store(false, std::memory_order_relaxed);
            busyFlag = false;
            q.clear();
            qSize.store(0, std::memory_order_relaxed);
        }
        th = std::thread([this] { this->run(); });
    }

    void requestStop() override {
        {
            std::lock_guard<std::mutex> lk(m);
            stop.store(true, std::memory_order_relaxed);
        }
        cvNotEmpty.notify_all();
        cvNotFull.notify_all();
        cvIdle.notify_all();
    }

    void join() override {
        if (th.joinable()) th.join();
    }

    ~UnifiedInferenceServerTrain() override {
        try {
            requestStop();
            join();
        }
        catch (...) {}
    }

    int size() const override {
        return qSize.load(std::memory_order_relaxed);
    }

    bool submit(std::unique_ptr<PendingNN>&& job,
        const std::atomic<bool>* extCancel = nullptr,
        const std::atomic<bool>* extAbort = nullptr) override {
        auto cancelled = [&]() -> bool {
            return stop.load(std::memory_order_relaxed) ||
                (extCancel && extCancel->load(std::memory_order_relaxed)) ||
                (extAbort && extAbort->load(std::memory_order_relaxed));
            };

        std::unique_lock<std::mutex> lk(m);

        while ((int)q.size() >= queueCap && !cancelled()) {
            cvNotFull.wait_for(lk, std::chrono::microseconds(AI_SUBMIT_WAIT_US));
        }

        if (cancelled()) return false;

        q.emplace_back(std::move(job));
        qSize.store((int)q.size(), std::memory_order_relaxed);

        lk.unlock();
        cvNotEmpty.notify_one();
        return true;
    }

    void waitIdle() override {
        std::unique_lock<std::mutex> lk(m);
        cvIdle.wait(lk, [&] {
            return q.empty() && !busyFlag;
            });
    }

    // Timed variant: returns false if the server did not go idle in time
    // (protects quiesce barriers from livelock under continuous submissions).
    bool waitIdleFor(int64_t timeoutMs) {
        std::unique_lock<std::mutex> lk(m);
        return cvIdle.wait_for(lk, std::chrono::milliseconds(timeoutMs), [&] {
            return q.empty() && !busyFlag;
            });
    }

    void clearQueueUnsafeWhenIdle() {
        std::deque<std::unique_ptr<PendingNN>> dropped;

        {
            std::unique_lock<std::mutex> lk(m);
            cvIdle.wait(lk, [&] { return !busyFlag; });

            dropped.swap(q);
            qSize.store((int)q.size(), std::memory_order_relaxed);
        }

        for (auto& p : dropped) {
            if (!p) continue;
            cancelPendingNN(*p);
            completePendingNNJob(*p);
            freePendingNN(std::move(p));
        }

        cvNotFull.notify_all();
        cvIdle.notify_all();
    }

private:
    AI_FORCEINLINE MCTSTable* resolveOwner(PendingNN& job) const noexcept {
        return job.ownerT ? job.ownerT : defaultOwner;
    }

    bool popBatchUnlocked(std::vector<std::unique_ptr<PendingNN>>& batch, int wantB) {
        batch.clear();
        batch.reserve((size_t)wantB);

        int n = 0;
        while (n < wantB && !q.empty()) {
            batch.emplace_back(std::move(q.front()));
            q.pop_front();
            ++n;
        }

        qSize.store((int)q.size(), std::memory_order_relaxed);
        return n != 0;
    }

    // IMPORTANT:
    // processBatch() only expands/cancels jobs.
    // Logical completion + reset + recycle are done by freePendingBatch() in the caller.
    void processBatch(std::vector<std::unique_ptr<PendingNN>>& jobs,
        std::vector<const PendingNN*>& batchPtrs,
        std::vector<float>& values
#if AI_HAVE_CUDA_KERNELS
        , std::vector<float>& logits
#else
        , std::vector<float>& policy
        , std::vector<Position>& posBatch
#endif
    ) {
        const int B = (int)jobs.size();
        if (B <= 0) return;
        recordInferBatchSize(B);

        batchPtrs.resize((size_t)B);
        for (int i = 0; i < B; ++i) batchPtrs[(size_t)i] = jobs[(size_t)i].get();

#if AI_HAVE_CUDA_KERNELS
        bool ok = false;
        {
            InferInFlightGuard ig;
            std::lock_guard<std::mutex> lk(backend.mtx);

            ok = backend.trt.inferBatchGather(batchPtrs.data(), B);
            if (ok) {
                backend.trt.copyValuesTo(values.data(), B);
                backend.trt.copyGatherLogitsTo(logits.data(), B);
            }
        }

        if (!ok) {
            {
                std::ostringstream oss;
                oss << "[UnifiedInferenceServerTrain] inferBatchGather failed, abort batch B=" << B;
                diagLogLine(oss.str());
            }
            for (int i = 0; i < B; ++i) {
                PendingNN& job = *jobs[(size_t)i];
                MCTSTable* owner = resolveOwner(job);
                if (owner) owner->abort.store(true, std::memory_order_release);
                cancelPendingNN(job);
            }
            return;
        }

        for (int i = 0; i < B; ++i) {
            PendingNN& job = *jobs[(size_t)i];
            MCTSTable* owner = resolveOwner(job);

            if (!owner || owner->abort.load(std::memory_order_relaxed)) {
                cancelPendingNN(job);
                continue;
            }

            float v = values[(size_t)i];
            const float* lg = logits.data() + (size_t)i * (size_t)AI_MAX_MOVES;

            expandLeafWithGatheredLogits(*owner, job, v, lg);
        }
#else
        posBatch.clear();
        posBatch.resize((size_t)B);
        for (int i = 0; i < B; ++i) posBatch[(size_t)i] = jobs[(size_t)i]->pos;

        bool ok = false;
        {
            InferInFlightGuard ig;
            std::lock_guard<std::mutex> lk(backend.mtx);

            ok = backend.trt.inferBatch(posBatch.data(), B);
            if (ok) {
                backend.trt.copyValuesTo(values.data(), B);
                backend.trt.copyPolicyTo(policy.data(), B);
            }
        }

        if (!ok) {
            {
                std::ostringstream oss;
                oss << "[UnifiedInferenceServerTrain] inferBatch failed, abort batch B=" << B;
                diagLogLine(oss.str());
            }
            for (int i = 0; i < B; ++i) {
                PendingNN& job = *jobs[(size_t)i];
                MCTSTable* owner = resolveOwner(job);
                if (owner) owner->abort.store(true, std::memory_order_release);
                cancelPendingNN(job);
            }
            return;
        }

        for (int i = 0; i < B; ++i) {
            PendingNN& job = *jobs[(size_t)i];
            MCTSTable* owner = resolveOwner(job);

            if (!owner || owner->abort.load(std::memory_order_relaxed)) {
                cancelPendingNN(job);
                continue;
            }

            float v = values[(size_t)i];
            const float* pol = policy.data() + (size_t)i * (size_t)POLICY_SIZE;

            expandLeafWithOutputs(*owner, job, v, pol);
        }
#endif
    }


    template<class TVec>
    void abortAndRecycleBatch(TVec& jobs) noexcept {
        for (auto& up : jobs) {
            if (!up) continue;

            PendingNN& job = *up;
            MCTSTable* owner = resolveOwner(job);
            if (owner) {
                owner->abort.store(true, std::memory_order_release);
            }

            cancelPendingNN(job);
        }

        freePendingBatch(jobs); // completion + reset + pool recycle exactly once
    }

    void emergencyAbortAndDrain(std::vector<std::unique_ptr<PendingNN>>& batch,
        std::vector<std::unique_ptr<PendingNN>>& add,
        const char* what) noexcept {
        try {
            diagLogLine(std::string("[UnifiedInferenceServerTrain] FATAL in run(): ")
                + (what ? what : "unknown exception"));
        }
        catch (...) {
        }

        stop.store(true, std::memory_order_relaxed);

        abortAndRecycleBatch(batch);
        abortAndRecycleBatch(add);

        std::vector<std::unique_ptr<PendingNN>> tail;
        try {
            std::lock_guard<std::mutex> lk(m);

            while (!q.empty()) {
                tail.emplace_back(std::move(q.front()));
                q.pop_front();
            }

            qSize.store(0, std::memory_order_relaxed);
            busyFlag = false;
        }
        catch (...) {
        }

        abortAndRecycleBatch(tail);

        cvNotEmpty.notify_all();
        cvNotFull.notify_all();
        cvIdle.notify_all();
    }

    void run() {
        std::vector<std::unique_ptr<PendingNN>> batch;
        std::vector<std::unique_ptr<PendingNN>> add;
        std::vector<const PendingNN*> batchPtrs;
        batch.reserve((size_t)TRT_MAX_BATCH);
        add.reserve((size_t)TRT_MAX_BATCH);

        std::vector<float> values((size_t)TRT_MAX_BATCH, 0.5f);

#if AI_HAVE_CUDA_KERNELS
        std::vector<float> logits((size_t)TRT_MAX_BATCH * (size_t)AI_MAX_MOVES, 0.0f);
#else
        std::vector<float> policy((size_t)TRT_MAX_BATCH * (size_t)POLICY_SIZE, 0.0f);
        std::vector<Position> posBatch;
        posBatch.reserve((size_t)TRT_MAX_BATCH);
#endif

        try {
            for (;;) {
                {
                    std::unique_lock<std::mutex> lk(m);

                    busyFlag = false;
                    if (q.empty()) cvIdle.notify_all();

                    cvNotEmpty.wait(lk, [&] {
                        return stop.load(std::memory_order_relaxed) || !q.empty();
                        });

                    if (stop.load(std::memory_order_relaxed) && q.empty()) break;

                    busyFlag = true;
                    (void)popBatchUnlocked(batch, TRT_MAX_BATCH);
                }

                cvNotFull.notify_all();

                const auto tFillEnd =
                    std::chrono::steady_clock::now() + std::chrono::microseconds(200);

                while ((int)batch.size() < TRT_MAX_BATCH &&
                    std::chrono::steady_clock::now() < tFillEnd) {
                    std::unique_lock<std::mutex> lk(m);

                    if (q.empty()) {
                        cvNotEmpty.wait_until(lk, tFillEnd, [&] {
                            return stop.load(std::memory_order_relaxed) || !q.empty();
                            });
                    }

                    if (q.empty()) break;

                    add.clear();
                    const int need = TRT_MAX_BATCH - (int)batch.size();
                    (void)popBatchUnlocked(add, need);
                    lk.unlock();

                    cvNotFull.notify_all();

                    for (auto& j : add) batch.emplace_back(std::move(j));
                }

#if AI_HAVE_CUDA_KERNELS
                const auto tBatchStart = std::chrono::steady_clock::now();
                processBatch(batch, batchPtrs, values, logits);
                recordInferBusyMicros((uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - tBatchStart).count());
#else
                const auto tBatchStart = std::chrono::steady_clock::now();
                processBatch(batch, batchPtrs, values, policy, posBatch);
                recordInferBusyMicros((uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - tBatchStart).count());
#endif
                freePendingBatch(batch);
            }

            for (;;) {
                std::vector<std::unique_ptr<PendingNN>> tail;
                {
                    std::lock_guard<std::mutex> lk(m);
                    if (q.empty()) break;
                    busyFlag = true;
                    (void)popBatchUnlocked(tail, TRT_MAX_BATCH);
                }

                cvNotFull.notify_all();

#if AI_HAVE_CUDA_KERNELS
                const auto tBatchStart = std::chrono::steady_clock::now();
                processBatch(tail, batchPtrs, values, logits);
                recordInferBusyMicros((uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - tBatchStart).count());
#else
                const auto tBatchStart = std::chrono::steady_clock::now();
                processBatch(tail, batchPtrs, values, policy, posBatch);
                recordInferBusyMicros((uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - tBatchStart).count());
#endif
                freePendingBatch(tail);
            }

            {
                std::lock_guard<std::mutex> lk(m);
                busyFlag = false;
                qSize.store((int)q.size(), std::memory_order_relaxed);
                if (q.empty()) cvIdle.notify_all();
            }

            cvNotFull.notify_all();
        }
        catch (const std::exception& e) {
            emergencyAbortAndDrain(batch, add, e.what());
        }
        catch (...) {
            emergencyAbortAndDrain(batch, add, "unknown exception");
        }
    }

};

struct InferenceServerTrain final : UnifiedInferenceServerTrain {
    explicit InferenceServerTrain(MCTSTable& tab, BackendBinding be)
        : UnifiedInferenceServerTrain(be, &tab, 8 * TRT_MAX_BATCH) {
    }
};

struct SharedInferenceServerTrain final : UnifiedInferenceServerTrain {
    explicit SharedInferenceServerTrain(BackendBinding be)
        : UnifiedInferenceServerTrain(be, nullptr, 16 * TRT_MAX_BATCH) {
    }
};
// ------------------------------------------------------------
// SearchPool: persistent MCTS workers (do NOT recreate threads for each search)
// ------------------------------------------------------------
static AI_FORCEINLINE bool tryClaimSimBudget(std::atomic<int>& simsLeft) {
    int cur = simsLeft.load(std::memory_order_relaxed);
    while (cur > 0) {
        if (simsLeft.compare_exchange_weak(
            cur, cur - 1,
            std::memory_order_relaxed,
            std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

static AI_FORCEINLINE void refundSimBudget(std::atomic<int>& simsLeft) {
    simsLeft.fetch_add(1, std::memory_order_relaxed);
}

struct SearchPoolStatsSnapshot {
    uint64_t simsOk = 0;
    uint64_t simsFail = 0;
    uint64_t ttHit = 0;
    uint64_t ttMiss = 0;
    uint64_t depthSum = 0;
};

struct SearchPool {
    std::vector<std::thread> pool;
    std::mutex m;
    std::condition_variable cv;
    std::mutex progressM;
    std::condition_variable cvProgress;

    bool stop = false;

    std::atomic<bool> cancelJob{ false };

    // FAIL-FAST state
    std::atomic<bool> fatal{ false };
    std::mutex fatalM;
    std::string fatalReason;

    // job dispatch
    int jobId = 0;
    std::atomic<int> workersBusy{ 0 };
    std::atomic<int> simsLeft{ 0 };
    std::atomic<uint64_t> activityTick{ 0 };
    std::atomic<uint64_t> progressTick{ 0 };
    std::atomic<uint64_t> statSimsOk{ 0 };
    std::atomic<uint64_t> statSimsFail{ 0 };
    std::atomic<uint64_t> statTTHit{ 0 };
    std::atomic<uint64_t> statTTMiss{ 0 };
    std::atomic<uint64_t> statDepthSum{ 0 };

    AI_FORCEINLINE void noteActivity() {
        const uint64_t t = activityTick.fetch_add(1, std::memory_order_relaxed) + 1ull;
        if ((t & 31ull) == 0ull) {
            cvProgress.notify_one();
        }
    }

    AI_FORCEINLINE void noteProgress() {
        noteActivity();
        const uint64_t t = progressTick.fetch_add(1, std::memory_order_relaxed) + 1ull;

        if ((t & 31ull) == 0ull) {
            cvProgress.notify_one();
        }
    }
    // job params (valid only during active job)
    MCTSTable* T = nullptr;
    ITrainInferenceServer* srv = nullptr;
    SearchWaitGroup* activeWG = nullptr;
    const Position* rootPos = nullptr;
    const std::array<uint64_t, 4>* path = nullptr;
    const std::array<int, 64>* mask = nullptr;
    SearchParams activeParams = kDefaultSearchParams;

    unsigned threads = 1;

    SearchPoolStatsSnapshot snapshotStats() const {
        SearchPoolStatsSnapshot s;
        s.simsOk = statSimsOk.load(std::memory_order_relaxed);
        s.simsFail = statSimsFail.load(std::memory_order_relaxed);
        s.ttHit = statTTHit.load(std::memory_order_relaxed);
        s.ttMiss = statTTMiss.load(std::memory_order_relaxed);
        s.depthSum = statDepthSum.load(std::memory_order_relaxed);
        return s;
    }

    void start(unsigned nThreads) {
        if (!pool.empty()) {
            throw std::logic_error("SearchPool::start() called while pool is already running");
        }

        const unsigned newThreads = std::max(1u, nThreads);
        std::vector<std::thread> newPool;
        newPool.reserve(newThreads);

        // Prepare clean "starting" state before workers begin.
        {
            std::lock_guard<std::mutex> lk(m);
            stop = false;

            // no active job at startup
            T = nullptr;
            srv = nullptr;
            rootPos = nullptr;
            path = nullptr;
            mask = nullptr;
            activeWG = nullptr;
            jobId = 0;
        }

        cancelJob.store(false, std::memory_order_relaxed);
        fatal.store(false, std::memory_order_relaxed);
        workersBusy.store(0, std::memory_order_relaxed);
        simsLeft.store(0, std::memory_order_relaxed);
        activityTick.store(0, std::memory_order_relaxed);
        progressTick.store(0, std::memory_order_relaxed);
        statSimsOk.store(0, std::memory_order_relaxed);
        statSimsFail.store(0, std::memory_order_relaxed);
        statTTHit.store(0, std::memory_order_relaxed);
        statTTMiss.store(0, std::memory_order_relaxed);
        statDepthSum.store(0, std::memory_order_relaxed);

        {
            std::lock_guard<std::mutex> lk(fatalM);
            fatalReason.clear();
        }

        try {
            for (unsigned tid = 0; tid < newThreads; ++tid) {
                newPool.emplace_back([this, tid] { this->workerMain(tid); });
            }
        }
        catch (...) {
            // Stop and wake any workers that were already created.
            {
                std::lock_guard<std::mutex> lk(m);
                stop = true;
            }

            cancelJob.store(true, std::memory_order_relaxed);
            simsLeft.store(0, std::memory_order_relaxed);
            cv.notify_all();
            cvProgress.notify_all();

            for (auto& th : newPool) {
                if (th.joinable()) th.join();
            }

            // Restore inert state.
            {
                std::lock_guard<std::mutex> lk(m);
                T = nullptr;
                srv = nullptr;
                rootPos = nullptr;
                path = nullptr;
                mask = nullptr;
                jobId = 0;
            }

            workersBusy.store(0, std::memory_order_relaxed);
            throw;
        }

        pool = std::move(newPool);
        threads = newThreads;
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lk(m);
            stop = true;
        }
        cancelJob.store(true, std::memory_order_relaxed);
        simsLeft.store(0, std::memory_order_relaxed);

        cv.notify_all();
        cvProgress.notify_all();

        for (auto& th : pool) {
            if (th.joinable()) th.join();
        }
        pool.clear();

        workersBusy.store(0, std::memory_order_relaxed);
        simsLeft.store(0, std::memory_order_relaxed);
    }

    ~SearchPool() {
        try { shutdown(); }
        catch (...) {}
    }

    bool isFatal() const {
        return fatal.load(std::memory_order_acquire);
    }

    std::string getFatalReason() const {
        std::lock_guard<std::mutex> lk(const_cast<std::mutex&>(fatalM));
        return fatalReason;
    }

    void requestFailFastNoThrow(const std::string& reason, MCTSTable* tt = nullptr) noexcept {
        bool wasFatal = fatal.exchange(true, std::memory_order_acq_rel);
        if (!wasFatal) {
            {
                std::lock_guard<std::mutex> lk(fatalM);
                fatalReason = reason;
            }
            diagLogLine(std::string("[SearchPool FATAL] ") + reason);
        }

        if (tt) {
            tt->abort.store(true, std::memory_order_release);
        }

        {
            std::lock_guard<std::mutex> lk(m);
            stop = true;
        }

        cancelJob.store(true, std::memory_order_relaxed);
        simsLeft.store(0, std::memory_order_relaxed);
        cv.notify_all();
        cvProgress.notify_all();
    }

    bool isPoolThreadId(std::thread::id id) const noexcept {
        for (const auto& th : pool) {
            if (th.joinable() && th.get_id() == id) return true;
        }
        return false;
    }

    void joinAllWorkersNoexcept() noexcept {
        const auto self = std::this_thread::get_id();

        for (auto& th : pool) {
            if (!th.joinable()) continue;
            if (th.get_id() == self) continue;

            try {
                th.join();
            }
            catch (...) {
            }
        }
    }

    void requestFailFast(const std::string& reason, MCTSTable* tt = nullptr) noexcept {
        requestFailFastNoThrow(reason, tt);
    }

    void failFast(const std::string& reason, MCTSTable* tt = nullptr) {
        requestFailFastNoThrow(reason, tt);

        // failFast() should throw only from the control thread.
        // If someone accidentally calls it from a worker thread, just do not throw:
        // worker should finish, and the control thread will see fatal and throw itself.
        if (isPoolThreadId(std::this_thread::get_id())) {
            return;
        }

        joinAllWorkersNoexcept();
        pool.clear();

        throw std::runtime_error("[SearchPool] FATAL: " + getFatalReason());
    }

    void runSims(MCTSTable& TT,
        ITrainInferenceServer& server,
        const Position& rp,
        const std::array<uint64_t, 4>& pth,
        const std::array<int, 64>& msk,
        int sims,
        const SearchParams& params = kDefaultSearchParams) {

        if (isFatal()) {
            throw std::runtime_error("[SearchPool] already failed: " + getFatalReason());
        }

        if (pool.empty()) {
            failFast("runSims() called with no worker threads", &TT);
        }

        if (TT.abort.load(std::memory_order_relaxed)) return;

        SearchWaitGroup wg;
        wg.pending.store(0, std::memory_order_relaxed);

        simsLeft.store(sims, std::memory_order_relaxed);
        cancelJob.store(false, std::memory_order_relaxed);
        workersBusy.store((int)threads, std::memory_order_relaxed);

        {
            std::lock_guard<std::mutex> lk(m);
            T = &TT;
            srv = &server;
            rootPos = &rp;
            path = &pth;
            mask = &msk;
            activeWG = &wg;
            activeParams = params;
            ++jobId;
        }
        cv.notify_all();
        cvProgress.notify_all();

        using Clock = std::chrono::steady_clock;

        // Watchdog thresholds:
        // warning only if nothing changed for a while,
        // fatal only if stall is really long.
        static constexpr auto WATCHDOG_WARN_AFTER = std::chrono::seconds(5);
        static constexpr auto WATCHDOG_FATAL_AFTER = std::chrono::seconds(30);

        uint64_t lastTick = progressTick.load(std::memory_order_relaxed);
        uint64_t lastActivity = activityTick.load(std::memory_order_relaxed);
        int lastSimsLeft = simsLeft.load(std::memory_order_relaxed);
        int lastQSize = server.size();
        int lastInFlight = g_inferInFlight.load(std::memory_order_relaxed);

        auto lastProgressAt = Clock::now();
        auto lastWarnAt = Clock::time_point::min();

        static constexpr auto WATCHDOG_WAIT_SLICE = std::chrono::milliseconds(250);

        for (;;) {
            if (isFatal()) {
                failFast(getFatalReason(), &TT);
            }

            if (workersBusy.load(std::memory_order_relaxed) == 0) break;

            {
                std::unique_lock<std::mutex> lk(progressM);
                cvProgress.wait_for(lk, WATCHDOG_WAIT_SLICE, [&] {
                    return isFatal() ||
                        workersBusy.load(std::memory_order_relaxed) == 0 ||
                        activityTick.load(std::memory_order_relaxed) != lastActivity ||
                        progressTick.load(std::memory_order_relaxed) != lastTick ||
                        simsLeft.load(std::memory_order_relaxed) != lastSimsLeft ||
                        server.size() != lastQSize ||
                        g_inferInFlight.load(std::memory_order_relaxed) != lastInFlight ||
                        TT.abort.load(std::memory_order_relaxed);
                    });
            }

            if (isFatal()) {
                failFast(getFatalReason(), &TT);
            }

            if (TT.abort.load(std::memory_order_relaxed)) {
                cancelJob.store(true, std::memory_order_relaxed);
                simsLeft.store(0, std::memory_order_relaxed);
            }

            const int busy = workersBusy.load(std::memory_order_relaxed);
            if (busy == 0) break;

            const auto now = Clock::now();

            const uint64_t tickNow = progressTick.load(std::memory_order_relaxed);
            const uint64_t activityNow = activityTick.load(std::memory_order_relaxed);
            const int simsNow = simsLeft.load(std::memory_order_relaxed);
            const int qNow = server.size();
            const int inFlightNow = g_inferInFlight.load(std::memory_order_relaxed);

            const bool progressed =
                (activityNow != lastActivity) ||
                (tickNow != lastTick) ||
                (simsNow != lastSimsLeft) ||
                (qNow != lastQSize) ||
                (inFlightNow != lastInFlight);

            if (progressed) {
                lastActivity = activityNow;
                lastTick = tickNow;
                lastSimsLeft = simsNow;
                lastQSize = qNow;
                lastInFlight = inFlightNow;
                lastProgressAt = now;
            }
            else {
                const auto stalledFor = now - lastProgressAt;

                if (stalledFor >= WATCHDOG_FATAL_AFTER) {
                    std::ostringstream oss;
                    oss << "stall watchdog fired: no progress for "
                        << std::chrono::duration_cast<std::chrono::milliseconds>(stalledFor).count()
                        << " ms"
                        << " busy=" << busy
                        << " simsLeft=" << simsNow
                        << " nnQueue=" << qNow
                        << " inferInFlight=" << inFlightNow
                        << " activityTick=" << activityNow
                        << " progressTick=" << tickNow
                        << " failGetNode=" << g_failGetNode.load(std::memory_order_relaxed)
                        << " failExpandWait=" << g_failExpandWait.load(std::memory_order_relaxed)
                        << " failDepth=" << g_failDepth.load(std::memory_order_relaxed)
                        << " ttAbort=" << TT.abort.load(std::memory_order_relaxed);
                    failFast(oss.str(), &TT);
                }

                if (stalledFor >= WATCHDOG_WARN_AFTER &&
                    (lastWarnAt == Clock::time_point::min() ||
                        now - lastWarnAt >= std::chrono::seconds(2))) {
                    lastWarnAt = now;
                    std::cerr << "[SearchPool] watchdog warning: stalled for "
                        << std::chrono::duration_cast<std::chrono::milliseconds>(stalledFor).count()
                        << " ms"
                        << " busy=" << busy
                        << " simsLeft=" << simsNow
                        << " nnQueue=" << qNow
                        << " inferInFlight=" << inFlightNow
                        << " activityTick=" << activityNow
                        << " progressTick=" << tickNow
                        << " failGetNode=" << g_failGetNode.load(std::memory_order_relaxed)
                        << " failExpandWait=" << g_failExpandWait.load(std::memory_order_relaxed)
                        << " failDepth=" << g_failDepth.load(std::memory_order_relaxed)
                        << " ttAbort=" << TT.abort.load(std::memory_order_relaxed)
                        << "\n";
                }
            }
        }

        // Escalating wait for outstanding NN jobs. A lost/stuck job used to hang
        // the whole training loop forever; now we log, then abort this search so
        // the server cancels its jobs (cancel path completes the wait group).
        {
            using WClock = std::chrono::steady_clock;
            const auto waitT0 = WClock::now();
            bool escalated = false;

            while (!waitGroupWaitZeroFor(&wg, std::chrono::seconds(30))) {
                const auto secs = std::chrono::duration_cast<std::chrono::seconds>(
                    WClock::now() - waitT0).count();

                std::cerr << "[runSims] waiting for NN jobs " << secs << "s"
                    << " pending=" << wg.pending.load(std::memory_order_relaxed)
                    << " nnQueue=" << server.size()
                    << " inFlight=" << g_inferInFlight.load(std::memory_order_relaxed)
                    << " ttAbort=" << TT.abort.load(std::memory_order_relaxed)
                    << std::endl;

                if (!escalated && secs >= 60) {
                    escalated = true;
                    std::cerr << "[runSims] escalating: aborting this search to release stuck jobs"
                        << std::endl;
                    cancelJob.store(true, std::memory_order_relaxed);
                    simsLeft.store(0, std::memory_order_relaxed);
                    TT.abort.store(true, std::memory_order_release);
                    cv.notify_all();
                    cvProgress.notify_all();
                }
            }
        }

        {
            std::lock_guard<std::mutex> lk(m);
            activeWG = nullptr;
        }

        if (isFatal()) {
            failFast(getFatalReason(), &TT);
        }
    }
private:
    void workerMain(unsigned tid) {
        int myJob = 0;
        uint32_t jitterBase = (uint32_t)(0x9E3779B9u * (tid + 1));

        for (;;) {
            MCTSTable* TT = nullptr;
            ITrainInferenceServer* server = nullptr;
            SearchWaitGroup* wg = nullptr;
            const Position* rp = nullptr;
            const std::array<uint64_t, 4>* pth = nullptr;
            const std::array<int, 64>* msk = nullptr;
            SearchParams paramsLocal = kDefaultSearchParams;

            bool busyAccounted = false;

            try {
                {
                    std::unique_lock<std::mutex> lk(m);
                    cv.wait(lk, [&] {
                        return stop || jobId != myJob;
                        });

                    if (stop) return;

                    myJob = jobId;
                    TT = T;
                    server = srv;
                    rp = rootPos;
                    pth = path;
                    msk = mask;
                    wg = activeWG;
                    paramsLocal = activeParams;
                }

                busyAccounted = true;

                if (!TT || TT->abort.load(std::memory_order_relaxed)) {
                    workersBusy.fetch_sub(1, std::memory_order_relaxed);
                    cvProgress.notify_all();
                    busyAccounted = false;
                    continue;
                }

                int k = 0;
                int queueSpins = 0;

                for (;;) {
                    if (fatal.load(std::memory_order_relaxed)) break;
                    if (TT->abort.load(std::memory_order_relaxed)) break;
                    if (cancelJob.load(std::memory_order_relaxed)) break;

                    if (server) {
                        throttleOnNNQueue_NoSleep(server->size(), queueSpins);

                        if (fatal.load(std::memory_order_relaxed)) break;
                        if (TT->abort.load(std::memory_order_relaxed)) break;
                        if (cancelJob.load(std::memory_order_relaxed)) break;
                    }

                    if (!tryClaimSimBudget(simsLeft)) break;

                    PendingNN localPending;
                    resetPendingNN(localPending);
                    PendingNNGuard localGuard(localPending);

                    bool needNN = false;
                    SimDiag sd{};

                    bool ok = runOneSim(*TT, *rp, *pth, *msk,
                        localPending, needNN,
                        jitterBase + (uint32_t)(k++) * 1337u,
                        paramsLocal,
                        &sd);

                    if (!ok) {
                        statSimsFail.fetch_add(1, std::memory_order_relaxed);
                        noteActivity();
                        refundSimBudget(simsLeft);

                        if (fatal.load(std::memory_order_relaxed)) break;
                        if (TT->abort.load(std::memory_order_relaxed)) break;
                        if (cancelJob.load(std::memory_order_relaxed)) break;

                        cpuRelax();
                        continue;
                    }

                    statSimsOk.fetch_add(1, std::memory_order_relaxed);
                    statTTHit.fetch_add(sd.ttHit, std::memory_order_relaxed);
                    statTTMiss.fetch_add(sd.ttMiss, std::memory_order_relaxed);
                    statDepthSum.fetch_add(sd.depth, std::memory_order_relaxed);

                    noteProgress();

                    if (needNN && server) {
                        throttleOnNNQueue_NoSleep(server->size(), queueSpins);

                        if (fatal.load(std::memory_order_relaxed) ||
                            cancelJob.load(std::memory_order_relaxed) ||
                            TT->abort.load(std::memory_order_relaxed)) {
                            break; // localGuard will cleanup
                        }

                        auto p = allocPendingNN();
                        PendingNNPtrGuard heapGuard(p);

                        *p = localPending;
                        localGuard.release();   // ownership moved from localPending to *p

                        p->ownerT = TT;
                        p->waitGroup = wg;

                        waitGroupAdd(wg);

                        if (!server->submit(std::move(p), &cancelJob, &TT->abort)) {
                            noteActivity();
                            if (!fatal.load(std::memory_order_relaxed) &&
                                !cancelJob.load(std::memory_order_relaxed) &&
                                !TT->abort.load(std::memory_order_relaxed)) {
                                refundSimBudget(simsLeft);
                            }
                            break; // heapGuard will cleanup + free
                        }

                        heapGuard.release();
                        noteProgress();
                    }
                    else if (needNN && !server) {
                        refundSimBudget(simsLeft);
                        break; // localGuard will cleanup
                    }
                    else {
                        localGuard.release(); // no pending NN ownership survived this iteration
                    }
                }

                workersBusy.fetch_sub(1, std::memory_order_relaxed);
                cvProgress.notify_all();
                busyAccounted = false;
            }
            catch (const std::exception& e) {
                if (busyAccounted) {
                    workersBusy.fetch_sub(1, std::memory_order_relaxed);
                    cvProgress.notify_all();
                    busyAccounted = false;
                }

                std::ostringstream oss;
                oss << "workerMain tid=" << tid << " exception: " << e.what();
                requestFailFast(oss.str(), TT);
                return;
            }
            catch (...) {
                if (busyAccounted) {
                    workersBusy.fetch_sub(1, std::memory_order_relaxed);
                    cvProgress.notify_all();
                    busyAccounted = false;
                }

                std::ostringstream oss;
                oss << "workerMain tid=" << tid << " unknown exception";
                requestFailFast(oss.str(), TT);
                return;
            }
        }
    }
};

// ------------------------------------------------------------
// Search fixed number of simulations (sims) with tree reuse
// Dirichlet noise is applied ONLY temporarily at root (does not permanently corrupt priors in TT)
// ------------------------------------------------------------

// Expand root (or any node keyed by rootPos) exactly once for training-selfplay.
// IMPORTANT:
//  - does NOT apply Dirichlet noise in permanent expansion
//  - marks GPU inference as "in flight" so trainer yields (InferInFlightGuard)
//  - protects TensorRT with g_trtMutex
static bool ensureExpandedTrain(MCTSTable& T,
    BackendBinding backend,
    const Position& rootPos,
    const std::array<uint64_t, 4>& path,
    const std::array<int, 64>& mask) {
    if (T.abort.load(std::memory_order_relaxed)) return false;

    TTNode* root = T.getNode(rootPos.key);
    if (!root) return false;

    // Reliably wait for / acquire root expansion.
    for (;;) {
        uint8_t ex = root->expanded.load(std::memory_order_acquire);

        if (ex == 1) return true;

        if (ex == 2) {
            if (!waitWhileExpanding(root)) {
                std::cerr << "[ensureExpandedTrain] timeout while waiting for root expansion, key="
                    << rootPos.key << "\n";
                T.abort.store(true, std::memory_order_release);
                return false;
            }
            continue; // reread root state
        }

        uint8_t expected = 0;
        if (root->expanded.compare_exchange_strong(expected, 2,
            std::memory_order_acq_rel,
            std::memory_order_relaxed)) {
            break; // we acquired expansion
        }

        // someone changed expanded first — try again
    }

    ExpansionClaimGuard rootClaim(root);

    MoveList ml;
    int term = 0;
    Position tmp = rootPos;

    genLegal(tmp, path, mask, ml, term);

    if (term) {
        root->key = rootPos.key;
        root->edgeBegin = 0;
        root->edgeCount = 0;
        root->terminal = 1;
        root->chance = 0;

        Trace empty; empty.reset();
        backprop(root, 1.0f, empty);
        publishTerminalWithMove(T, root, rootPos.key, ml.n ? ml.m[0] : 0);
        rootClaim.release();
        return true;
    }

    if (ml.n == 0) {
        publishReady(root, rootPos.key, 0, 0, 0, 1);
        rootClaim.release();
        return true;
    }

    PendingNN p;
    resetPendingNN(p);
    PendingNNGuard pGuard(p);

    p.leaf = root;
    p.pos = rootPos;
    p.ml = ml;
    p.trace.reset();
    fillPendingPolicyIdx(p);

    rootClaim.release();
    float v = 0.5f;

#if AI_HAVE_CUDA_KERNELS
    std::array<float, AI_MAX_MOVES> logitsLocal{};
    bool ok = false;

    {
        InferInFlightGuard ig;
        std::lock_guard<std::mutex> lk(backend.mtx);

        ok = backend.trt.inferBatchGather(&p, 1);
        if (ok) {
            backend.trt.copyValuesTo(&v, 1);
            backend.trt.copyGatherLogitsTo(logitsLocal.data(), 1);
        }
    }

    if (!ok) {
        {
            std::ostringstream oss;
            oss << "[ensureExpandedTrain] inferBatchGather failed for root key=" << rootPos.key;
            diagLogLine(oss.str());
        }
        T.abort.store(true, std::memory_order_release);
        return false; // pGuard will cleanup
    }

    expandLeafWithGatheredLogits(T, p, v, logitsLocal.data());
    pGuard.release();
#else
    std::vector<float> pol((size_t)POLICY_SIZE, 0.0f);
    bool ok = false;

    {
        InferInFlightGuard ig;
        std::lock_guard<std::mutex> lk(backend.mtx);
        ok = backend.trt.inferBatch(&p.pos, 1, &v, pol.data());
    }

    if (!ok) {
        {
            std::ostringstream oss;
            oss << "[ensureExpandedTrain] inferBatch failed for root key=" << rootPos.key;
            diagLogLine(oss.str());
        }
        T.abort.store(true, std::memory_order_release);
        return false; // pGuard will cleanup
    }

    expandLeafWithOutputs(T, p, v, pol.data());
    pGuard.release();
#endif

    return root->expanded.load(std::memory_order_acquire) == 1;
}
static void collectRootMoves(MCTSTable& T,
    const Position& rootPos,
    float& outQSideToMove,
    std::vector<moveState>& outMoves) {
    TTNode* root = T.findNodeNoInsert(rootPos.key);
    if (!root) { outQSideToMove = 0.5f; outMoves.clear(); return; }

    outQSideToMove = nodeQ(*root);

    outMoves.clear();
    uint8_t ex = root->expanded.load(std::memory_order_acquire);
    if (ex != 1 || !root->edgeCount) return;

    TTEdge* e0 = T.edgePtr(root->edgeBegin);
    outMoves.reserve(root->edgeCount);

    for (int i = 0; i < (int)root->edgeCount; ++i) {
        const TTEdge& e = e0[i];
        uint32_t v = e.visits.load(std::memory_order_relaxed);

        float ev = -1.0f;
        if (v) ev = clamp01((float)(e.sum() / (double)v));

        outMoves.push_back(moveState{ e.move, ev, v, e.prior(), 0ull, {} });
    }

    std::sort(outMoves.begin(), outMoves.end(),
        [](const moveState& a, const moveState& b) {
            if (a.visits != b.visits) return a.visits > b.visits;
            return a.eval > b.eval;
        });
}

static int pickMoveFromVisits(const std::vector<moveState>& mv, float temperature) {
    if (mv.empty()) return 0;

    if (!(temperature > 1e-6f)) return mv[0].move;

    const double invTemp = 1.0 / (double)temperature;

    double sum = 0.0;
    for (size_t i = 0; i < mv.size(); ++i) {
        const double v = (double)mv[i].visits;
        sum += std::pow(v + 1e-9, invTemp);
    }

    if (!(sum > 0.0)) return mv[0].move;

    std::uniform_real_distribution<double> d(0.0, sum);
    double r = d(Random);

    double acc = 0.0;
    for (size_t i = 0; i < mv.size(); ++i) {
        const double v = (double)mv[i].visits;
        acc += std::pow(v + 1e-9, invTemp);
        if (r <= acc) return mv[i].move;
    }

    return mv.back().move;
}

// policy target — SPARSE (idx/prob), idx in CHW: k=pl*64+sq
static void buildSparsePolicyTargetCHW(const Position& pos,
    const std::vector<moveState>& mv,
    uint16_t& outN,
    std::array<uint16_t, AI_MAX_MOVES>& outIdx,
    std::array<uint16_t, AI_MAX_MOVES>& outProbQ) {
    outN = 0;
    outIdx.fill(0);
    outProbQ.fill(0);

    if (mv.empty()) return;

    const int n = std::min((int)mv.size(), AI_MAX_MOVES);

    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += (double)mv[(size_t)i].visits;
    }

    outN = (uint16_t)n;

    if (!(sum > 0.0)) {
        const float inv = 1.0f / (float)n;
        for (int i = 0; i < n; ++i) {
            int k = policyIndexCHWCanonical(mv[(size_t)i].move, pos);
            outIdx[(size_t)i] = (uint16_t)k;
            outProbQ[(size_t)i] = quantizeProbU16(inv);
        }
        return;
    }

    const float inv = (float)(1.0 / sum);
    for (int i = 0; i < n; ++i) {
        int k = policyIndexCHWCanonical(mv[(size_t)i].move, pos);
        float p = (float)mv[(size_t)i].visits * inv;

        outIdx[(size_t)i] = (uint16_t)k;
        outProbQ[(size_t)i] = quantizeProbU16(p);
    }
}

// temporarily (for one search) perturb root priors, then roll back
static void runFixedSims(MCTSTable& T,
    SearchPool& pool,
    ITrainInferenceServer& srv,
    BackendBinding backend,
    const Position& rootPos,
    const std::array<uint64_t, 4>& path,
    const std::array<int, 64>& mask,
    int sims,
    bool rootNoise,
    const SearchParams& params = kDefaultSearchParams) {
    if (T.abort.load(std::memory_order_relaxed)) return;

    if (!ensureExpandedTrain(T, backend, rootPos, path, mask)) return;
    if (T.abort.load(std::memory_order_relaxed)) return;

    RootNoiseGuard rootNoiseGuard(T, rootPos, rootNoise);

    pool.runSims(T, srv, rootPos, path, mask, sims, params);
}
// ------------------------------------------------------------
// Self-play: reuse one MCTSTable + one InferenceServerTrain + SearchPool
// ------------------------------------------------------------

static AI_FORCEINLINE void resetMCTSTableForNewGame(MCTSTable& T) {
    // O(1) reset via generation counter
    T.newGame();
}

struct GameContext {
    MCTSTable T;
    SearchPool pool;

    explicit GameContext(size_t nodePow2, size_t edgeCap)
        : T(nodePow2, edgeCap) {
    }

    void start(unsigned forcedThreads = 0) {
        unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        unsigned n = forcedThreads ? forcedThreads : std::min(hw, 4u);
        pool.start(n);
    }

    void stop() {
        pool.shutdown();
    }

    void resetForNewGame() {
        resetMCTSTableForNewGame(T);
    }

    ~GameContext() {
        try { stop(); }
        catch (...) {}
    }
};

struct SelfPlayContext {
    MCTSTable T;
    BackendBinding backend;
    InferenceServerTrain server;
    SearchPool pool;

    explicit SelfPlayContext(size_t nodePow2, size_t edgeCap,
        TrtRunner& trt, std::mutex& mtx)
        : T(nodePow2, edgeCap)
        , backend{ trt, mtx }
        , server(T, backend) {
    }

    void start(unsigned forcedThreads = 0) {
        server.start();
        unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        unsigned n = forcedThreads ? forcedThreads : std::min(hw, 8u);
        pool.start(n);
    }

    void stop() {
        pool.shutdown();
        server.requestStop();
        server.join();
    }

    void resetForNewGame() {
        server.waitIdle();
        server.clearQueueUnsafeWhenIdle();
        resetMCTSTableForNewGame(T);
    }

    ~SelfPlayContext() {
        try { stop(); }
        catch (...) {}
    }
};

struct ArenaStats {
    int curWins = 0;
    int oldWins = 0;
    int draws = 0;

    double currentScore() const {
        int n = curWins + oldWins;
        if (n <= 0) return 0.5;
        return (double)curWins / (double)n;
    }
};

struct MatchStatsGeneric {
    int p1Wins = 0;
    int p2Wins = 0;
    int draws = 0;

    double p1Score() const {
        const int n = p1Wins + p2Wins;
        if (n <= 0) return 0.5;
        return (double)p1Wins / (double)n;
    }
};

template<class Lane>
struct LanesStopGuard {
    std::vector<std::unique_ptr<Lane>>* lanes = nullptr;

    explicit LanesStopGuard(std::vector<std::unique_ptr<Lane>>& v) : lanes(&v) {}

    ~LanesStopGuard() noexcept {
        if (!lanes) return;
        for (auto& x : *lanes) {
            if (!x) continue;
            try { x->stop(); }
            catch (...) {}
        }
    }

    void release() noexcept { lanes = nullptr; }

    LanesStopGuard(const LanesStopGuard&) = delete;
    LanesStopGuard& operator=(const LanesStopGuard&) = delete;
};

template<class FindChanceNodeFn, class SearchMovesFn>
static int playOneUniversalMatchGame(
    const Position& startPos,
    const std::array<uint64_t, 4>& path,
    const std::array<int, 64>& mask,
    bool p1IsWhite,
    int maxPlies,
    FindChanceNodeFn&& findChanceNodeForSide,
    SearchMovesFn&& searchMovesForSide,
    const std::vector<int>* mirroredDice = nullptr,
    std::vector<int>* producedDice = nullptr)
{
    Position pos = startPos;
    size_t chanceIdx = 0;

    for (int ply = 0; ply < maxPlies; ++ply) {
        MoveList ml;
        int term = 0;
        Position tmp = pos;
        genLegal(tmp, path, mask, ml, term);

        if (term) {
            const bool p1Won = ((pos.side == 0) == p1IsWhite);
            return p1Won ? +1 : -1;
        }

        const bool p1Turn = ((pos.side == 0) == p1IsWhite);

        if (ml.n == 0) {
            TTNode* n = findChanceNodeForSide(p1Turn, pos);
            if (mirroredDice && chanceIdx < mirroredDice->size()) {
                makeRandomWithRolledDice(pos, n, (*mirroredDice)[chanceIdx]);
            }
            else {
                const int rolledDice = Dice[Range(Random)];
                if (producedDice) producedDice->push_back(rolledDice);
                makeRandomWithRolledDice(pos, n, rolledDice);
            }
            ++chanceIdx;
            continue;
        }

        std::vector<moveState> moves;
        if (!searchMovesForSide(p1Turn, pos, path, mask, moves)) {
            return 0;
        }

        if (moves.empty()) return 0;

        const int mv = moves[0].move; // temperature=0 for match play
        if (!mv) return 0;

        makeMove(pos, mask, mv);
    }

    return 0; // draw by maxPlies
}

template<class Lane, class PlayOneFn, class ProgressFn>
static MatchStatsGeneric runUniversalMatchEngine(
    std::vector<std::unique_ptr<Lane>>& lanes,
    int games,
    PlayOneFn&& playOneOnLane,
    ProgressFn&& onProgress,
    int progressEveryPairs = 0)
{
    MatchStatsGeneric out{};
    if (games <= 0 || lanes.empty()) return out;

    const int pairs = games / 2;

    std::atomic<int> nextPair{ 0 };

    std::atomic<bool> abortAll{ false };

    std::mutex exM;
    std::exception_ptr ex;

    std::mutex printM;
    std::vector<std::thread> outer;
    outer.reserve(lanes.size());

    // Result counters live under resM together with the report trigger, so every
    // progress line covers EXACTLY reportEvery new games. (Reporting off a
    // separate pair counter used to race with other lanes: a "100 games" line
    // could show 103 decided games.)
    //
    // Both games of a pair (same opening + dice, colors swapped) are recorded in
    // ONE locked update, so a report can never split a pair: it always contains
    // both games of every pair it covers, or neither. Splitting would bias the
    // line, because the two games of a pair are strongly correlated.
    std::mutex resM;
    int w1 = 0, w2 = 0, dr = 0;
    const int reportEvery = (progressEveryPairs > 0) ? (progressEveryPairs * 2) : 0;

    auto addResults = [&](int r1, bool hasSecond, int r2) {
        auto applyOne = [&](int r) {
            if (r > 0) ++w1;
            else if (r < 0) ++w2;
            else ++dr;
            };

        MatchStatsGeneric snap;
        int total = 0;
        bool report = false;
        {
            std::lock_guard<std::mutex> lk(resM);
            applyOne(r1);
            if (hasSecond) applyOne(r2);

            total = w1 + w2 + dr;
            if (reportEvery > 0 && (total % reportEvery) == 0) {
                snap.p1Wins = w1;
                snap.p2Wins = w2;
                snap.draws = dr;
                report = true;
            }
        }
        if (report) {
            std::lock_guard<std::mutex> lk(printM);
            onProgress(total, snap);
        }
        };

    auto snapshotNow = [&](int& outTotal) {
        MatchStatsGeneric snap;
        std::lock_guard<std::mutex> lk(resM);
        snap.p1Wins = w1;
        snap.p2Wins = w2;
        snap.draws = dr;
        outTotal = w1 + w2 + dr;
        return snap;
        };

    for (size_t li = 0; li < lanes.size(); ++li) {
        outer.emplace_back([&, li] {
            try {
                Lane& lane = *lanes[li];

                for (;;) {
                    if (abortAll.load(std::memory_order_relaxed)) break;

                    const int pairIdx = nextPair.fetch_add(1, std::memory_order_relaxed);
                    if (pairIdx >= pairs) break;

                    Position startPos;
                    std::array<uint64_t, 4> path;
                    std::array<int, 64> mask;
                    chess960(startPos, path, mask);

                    std::vector<int> firstGameDice;
                    lane.resetForNewGame();
                    const int r1 = playOneOnLane(lane, startPos, path, mask, /*p1IsWhite=*/true, nullptr, &firstGameDice);

                    if (abortAll.load(std::memory_order_relaxed)) {
                        // Aborted mid-pair: drop this orphan game. It was played
                        // with one color only, so counting it would bias W/L, and
                        // an odd increment would break the pair-aligned reporting
                        // (totals would never hit a multiple of reportEvery again).
                        (void)r1;
                        break;
                    }

                    lane.resetForNewGame();
                    const int r2 = playOneOnLane(lane, startPos, path, mask, /*p1IsWhite=*/false, &firstGameDice, nullptr);

                    addResults(r1, /*hasSecond=*/true, r2); // whole pair recorded at once
                }
            }
            catch (...) {
                abortAll.store(true, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lk(exM);
                if (!ex) ex = std::current_exception();
            }
            });
    }

    for (auto& th : outer) {
        if (th.joinable()) th.join();
    }

    if (ex) std::rethrow_exception(ex);

    if ((games & 1) != 0 && !lanes.empty()) {
        Position startPos;
        std::array<uint64_t, 4> path;
        std::array<int, 64> mask;
        chess960(startPos, path, mask);

        Lane& lane = *lanes[0];
        std::vector<int> firstGameDice;
        lane.resetForNewGame();
        addResults(playOneOnLane(lane, startPos, path, mask, /*p1IsWhite=*/true, nullptr, &firstGameDice),
            /*hasSecond=*/false, 0);

        // Tail game: report only if addResult() did not already do it.
        int tailTotal = 0;
        MatchStatsGeneric tailSnap = snapshotNow(tailTotal);
        if (reportEvery > 0 && (tailTotal % reportEvery) != 0) {
            onProgress(tailTotal, tailSnap);
        }
    }

    int finalTotal = 0;
    out = snapshotNow(finalTotal);
    return out;
}

struct ArenaLane {
    SelfPlayContext curCtx;
    SelfPlayContext oldCtx;

    ArenaLane()
        : curCtx((1u << 19), (1u << 23), g_trt, g_trtMutex)
        , oldCtx((1u << 19), (1u << 23), g_trt_old, g_trtOldMutex) {
    }

    void start(unsigned threadsPerSide) {
        curCtx.start(threadsPerSide);
        oldCtx.start(threadsPerSide);
    }

    void stop() {
        curCtx.stop();
        oldCtx.stop();
    }

    void resetForNewGame() {
        curCtx.resetForNewGame();
        oldCtx.resetForNewGame();
    }
};

static int playOneArenaGameOnLane(
    ArenaLane& lane,
    const Position& startPos,
    const std::array<uint64_t, 4>& path,
    const std::array<int, 64>& mask,
    bool currentIsWhite,
    int simsPerPos,
    int maxPlies = 256,
    const std::vector<int>* mirroredDice = nullptr,
    std::vector<int>* producedDice = nullptr)
{
    auto findChanceNode = [&](bool currentTurn, const Position& pos) -> TTNode* {
        return currentTurn
            ? lane.curCtx.T.findNodeNoInsert(pos.key)
            : lane.oldCtx.T.findNodeNoInsert(pos.key);
        };

    auto searchMoves = [&](bool currentTurn,
        const Position& pos,
        const std::array<uint64_t, 4>& pathRef,
        const std::array<int, 64>& maskRef,
        std::vector<moveState>& moves) -> bool {
            SelfPlayContext& ctx = currentTurn ? lane.curCtx : lane.oldCtx;

            float q = 0.5f;
            runFixedSims(ctx.T, ctx.pool, ctx.server, ctx.backend,
                pos, pathRef, maskRef, simsPerPos, /*rootNoise=*/false);

            if (ctx.T.abort.load(std::memory_order_relaxed)) {
                return false;
            }

            collectRootMoves(ctx.T, pos, q, moves);
            return !moves.empty();
        };

    return playOneUniversalMatchGame(
        startPos, path, mask, currentIsWhite, maxPlies,
        findChanceNode, searchMoves, mirroredDice, producedDice);
}

static double computeLOSPercent(int wins, int losses);

static ArenaStats runArenaMatch(int games, int simsPerPos) {
    ArenaStats st;
    if (games <= 0) return st;

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());

    // In arena, each lane = 2 SelfPlayContext, each with its own server + pool.
    // Therefore keep threadsPerSide small.
    const unsigned threadsPerSide = 1;
    const unsigned wantedLanes = (unsigned)std::max(1, (games + 1) / 2);
    const unsigned lanesByFormula = (hw > 4u) ? ((hw - 4u) / 2u) : 1u;
    const unsigned parallelLanes = std::max(1u, std::min(wantedLanes, lanesByFormula));

    std::vector<std::unique_ptr<ArenaLane>> lanes;
    lanes.reserve(parallelLanes);
    LanesStopGuard<ArenaLane> guard(lanes);

    for (unsigned i = 0; i < parallelLanes; ++i) {
        auto lane = std::make_unique<ArenaLane>();
        lane->start(threadsPerSide);
        lanes.push_back(std::move(lane));
    }

    auto onProgress = [&](int playedGames, const MatchStatsGeneric& s) {
        noteTrainingProgress();
        if ((playedGames % 2) == 0 && (playedGames % 100) == 0) {
            const double los = computeLOSPercent(s.p1Wins, s.p2Wins);
            std::cout << "[arena] games=" << playedGames
                << " W/L=" << s.p1Wins << "/" << s.p2Wins
                << " score=" << std::fixed << std::setprecision(4) << s.p1Score()
                << " LOS=" << std::setprecision(2) << los << "%" << std::endl;
        }
        };

    MatchStatsGeneric g = runUniversalMatchEngine(
        lanes,
        games,
        [&](ArenaLane& lane,
            const Position& startPos,
            const std::array<uint64_t, 4>& path,
            const std::array<int, 64>& mask,
            bool p1IsWhite,
            const std::vector<int>* mirroredDice,
            std::vector<int>* producedDice) -> int {
                return playOneArenaGameOnLane(
                    lane, startPos, path, mask, p1IsWhite, simsPerPos, 256, mirroredDice, producedDice);
        },
        onProgress,
        /*progressEveryPairs=*/50);

    st.curWins = g.p1Wins;
    st.oldWins = g.p2Wins;
    st.draws = g.draws;
    return st;
}

static AI_FORCEINLINE double normalCdf(double z) {
    return 0.5 * std::erfc(-z / std::sqrt(2.0));
}

static double computeLOSPercent(int wins, int losses) {
    const int n = wins + losses;
    if (n <= 0) return 50.0;

    const double mean = (double)wins / (double)n;
    const double ex2 = (double)wins / (double)n;
    double var = ex2 - mean * mean;
    if (var < 0.0) var = 0.0;

    if (var <= 1e-15) {
        if (mean > 0.5) return 100.0;
        if (mean < 0.5) return 0.0;
        return 50.0;
    }

    const double se = std::sqrt(var / (double)n);
    const double z = (mean - 0.5) / se;
    return 100.0 * normalCdf(z);
}

static void printTuneProgress(int played, int wins1, int losses1) {
    const int n = wins1 + losses1;
    const double score1 = (n > 0)
        ? ((double)wins1 / (double)n)
        : 0.5;

    const double los = computeLOSPercent(wins1, losses1);

    std::cout
        << "[tune] games=" << played
        << " W/L=" << wins1 << "/" << losses1
        << " score=" << std::fixed << std::setprecision(4) << score1
        << " LOS=" << std::setprecision(2) << los << '%' << std::endl;
}

struct TuneLane {
    GameContext p1Ctx;
    GameContext p2Ctx;

    TuneLane()
        : p1Ctx((1u << 19), (1u << 23))
        , p2Ctx((1u << 19), (1u << 23)) {
    }

    void start(unsigned threadsPerSide) {
        p1Ctx.start(threadsPerSide);
        p2Ctx.start(threadsPerSide);
    }

    void stop() {
        p1Ctx.stop();
        p2Ctx.stop();
    }

    void resetForNewGame() {
        p1Ctx.resetForNewGame();
        p2Ctx.resetForNewGame();
    }
};

static int playOneTuneGameOnLane(
    TuneLane& lane,
    ITrainInferenceServer& sharedSrv,
    BackendBinding backend,
    const SearchParams& p1,
    const SearchParams& p2,
    const Position& startPos,
    const std::array<uint64_t, 4>& path,
    const std::array<int, 64>& mask,
    bool p1IsWhite,
    int simsPerPos,
    int maxPlies = 256,
    const std::vector<int>* mirroredDice = nullptr,
    std::vector<int>* producedDice = nullptr)
{
    auto findChanceNode = [&](bool p1Turn, const Position& pos) -> TTNode* {
        return p1Turn
            ? lane.p1Ctx.T.findNodeNoInsert(pos.key)
            : lane.p2Ctx.T.findNodeNoInsert(pos.key);
        };

    auto searchMoves = [&](bool p1Turn,
        const Position& pos,
        const std::array<uint64_t, 4>& pathRef,
        const std::array<int, 64>& maskRef,
        std::vector<moveState>& moves) -> bool {
            GameContext& ctx = p1Turn ? lane.p1Ctx : lane.p2Ctx;
            const SearchParams& sp = p1Turn ? p1 : p2;

            float q = 0.5f;
            runFixedSims(ctx.T, ctx.pool, sharedSrv, backend,
                pos, pathRef, maskRef, simsPerPos, /*rootNoise=*/false, sp);

            if (ctx.T.abort.load(std::memory_order_relaxed)) {
                return false;
            }

            collectRootMoves(ctx.T, pos, q, moves);
            return !moves.empty();
        };

    return playOneUniversalMatchGame(
        startPos, path, mask, p1IsWhite, maxPlies,
        findChanceNode, searchMoves, mirroredDice, producedDice);
}

struct NetArenaLane {
    GameContext n1Ctx;
    GameContext n2Ctx;

    NetArenaLane()
        : n1Ctx((1u << 19), (1u << 23))
        , n2Ctx((1u << 19), (1u << 23)) {
    }

    void start(unsigned threadsPerSide) {
        n1Ctx.start(threadsPerSide);
        n2Ctx.start(threadsPerSide);
    }

    void stop() {
        n1Ctx.stop();
        n2Ctx.stop();
    }

    void resetForNewGame() {
        n1Ctx.resetForNewGame();
        n2Ctx.resetForNewGame();
    }
};

static int playOneNetArenaGameOnLane(
    NetArenaLane& lane,
    ITrainInferenceServer& n1Srv,
    ITrainInferenceServer& n2Srv,
    BackendBinding n1Backend,
    BackendBinding n2Backend,
    const SearchParams& n1Params,
    const SearchParams& n2Params,
    const Position& startPos,
    const std::array<uint64_t, 4>& path,
    const std::array<int, 64>& mask,
    bool n1IsWhite,
    int simsPerPos,
    int maxPlies = 256,
    const std::vector<int>* mirroredDice = nullptr,
    std::vector<int>* producedDice = nullptr)
{
    auto findChanceNode = [&](bool n1Turn, const Position& pos) -> TTNode* {
        return n1Turn
            ? lane.n1Ctx.T.findNodeNoInsert(pos.key)
            : lane.n2Ctx.T.findNodeNoInsert(pos.key);
        };

    auto searchMoves = [&](bool n1Turn,
        const Position& pos,
        const std::array<uint64_t, 4>& pathRef,
        const std::array<int, 64>& maskRef,
        std::vector<moveState>& moves) -> bool {
            GameContext& ctx = n1Turn ? lane.n1Ctx : lane.n2Ctx;
            ITrainInferenceServer& srv = n1Turn ? n1Srv : n2Srv;
            BackendBinding backend = n1Turn ? n1Backend : n2Backend;
            const SearchParams& sp = n1Turn ? n1Params : n2Params;

            float q = 0.5f;
            runFixedSims(ctx.T, ctx.pool, srv, backend,
                pos, pathRef, maskRef, simsPerPos, /*rootNoise=*/false, sp);

            if (ctx.T.abort.load(std::memory_order_relaxed)) {
                return false;
            }

            collectRootMoves(ctx.T, pos, q, moves);
            return !moves.empty();
        };

    return playOneUniversalMatchGame(
        startPos, path, mask, n1IsWhite, maxPlies,
        findChanceNode, searchMoves, mirroredDice, producedDice);
}

void arena(string net1, string net2) {

    TrtRunner trt1;
    TrtRunner trt2;
    std::mutex trt1Mutex;
    std::mutex trt2Mutex;
    if (!trt1.initOrCreate(net1)) {
        std::cerr << "[arena-net] failed to initialize net1: " << net1 << "\n";

        return;
    }
    if (!trt2.initOrCreate(net2)) {
        std::cerr << "[arena-net] failed to initialize net2: " << net2 << "\n";

        trt1.shutdown();

        return;
    }

    BackendBinding n1Backend{ trt1, trt1Mutex };
    BackendBinding n2Backend{ trt2, trt2Mutex };
    SharedInferenceServerTrain n1Srv(n1Backend);
    SharedInferenceServerTrain n2Srv(n2Backend);
    n1Srv.start();
    n2Srv.start();

    struct NetArenaCleanupGuard {
        SharedInferenceServerTrain* n1Srv = nullptr;
        SharedInferenceServerTrain* n2Srv = nullptr;
        TrtRunner* trt1 = nullptr;
        TrtRunner* trt2 = nullptr;
        ~NetArenaCleanupGuard() noexcept {
            try {
                if (n1Srv) {
                    n1Srv->requestStop();
                    n1Srv->join();
                }
            }
            catch (...) {}
            try {
                if (n2Srv) {
                    n2Srv->requestStop();
                    n2Srv->join();
                }
            }
            catch (...) {}
            try {
                if (trt1) trt1->shutdown();
                if (trt2) trt2->shutdown();
            }
            catch (...) {}
        }
    } guard{ &n1Srv, &n2Srv, &trt1, &trt2 };

    const SearchParams n1Params = kDefaultSearchParams;
    const SearchParams n2Params = kDefaultSearchParams;

    // Practically unlimited: an external sequential test decides when to stop.
    static constexpr int TOTAL_GAMES = 2000000;
    static constexpr int SIMS_PER_POS = 800;
    static constexpr int MAX_PLIES = 256;

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const unsigned threadsPerSide = 1;
    const unsigned wantedLanes = (unsigned)std::max(1, (TOTAL_GAMES + 1) / 2);
    const unsigned lanesByFormula = (hw > 4u) ? ((hw - 4u) / 2u) : 1u;
    const unsigned parallelLanes = std::max(1u, std::min(wantedLanes, lanesByFormula));

    std::vector<std::unique_ptr<NetArenaLane>> lanes;
    lanes.reserve(parallelLanes);
    LanesStopGuard<NetArenaLane> lanesGuard(lanes);

    for (unsigned i = 0; i < parallelLanes; ++i) {
        auto lane = std::make_unique<NetArenaLane>();
        lane->start(threadsPerSide);
        lanes.push_back(std::move(lane));
    }

    std::cout
        << "[arena-net] start\n"
        << "  net1: " << net1 << "\n"
        << "  net2: " << net2 << "\n"
        << "  games=" << TOTAL_GAMES << " sims=" << SIMS_PER_POS << "\n"
        << "  parallel_lanes=" << parallelLanes
        << " threads_per_side=" << threadsPerSide << std::endl;

    auto onProgress = [&](int playedGames, const MatchStatsGeneric& s) {
        if ((playedGames % 2) == 0 && (playedGames % 100) == 0) {
            printTuneProgress(playedGames, s.p1Wins, s.p2Wins);
        }
        };

    MatchStatsGeneric g = runUniversalMatchEngine(
        lanes,
        TOTAL_GAMES,
        [&](NetArenaLane& lane,
            const Position& startPos,
            const std::array<uint64_t, 4>& path,
            const std::array<int, 64>& mask,
            bool n1IsWhite,
            const std::vector<int>* mirroredDice,
            std::vector<int>* producedDice) -> int {
                return playOneNetArenaGameOnLane(
                    lane,
                    n1Srv,
                    n2Srv,
                    n1Backend,
                    n2Backend,
                    n1Params,
                    n2Params,
                    startPos,
                    path,
                    mask,
                    n1IsWhite,
                    SIMS_PER_POS,
                    MAX_PLIES,
                    mirroredDice,
                    producedDice
                );
        },
        onProgress,
        /*progressEveryPairs=*/50);

    printTuneProgress(TOTAL_GAMES, g.p1Wins, g.p2Wins);
    std::cout << "[arena-net] finished\n";
}

void tune(float c_init1, float fpu_reduction1,
    float c_init2, float fpu_reduction2) {
    if (!g_trtReady) {
        std::cerr << "[tune] TensorRT backend is not ready.\n";
        return;
    }

    const SearchParams p1 = makeSearchParams(c_init1, fpu_reduction1);
    const SearchParams p2 = makeSearchParams(c_init2, fpu_reduction2);

    static constexpr int TOTAL_GAMES = 10000;
    static constexpr int SIMS_PER_POS = 800;
    static constexpr int MAX_PLIES = 256;

    BackendBinding backend{ g_trt, g_trtMutex };
    SharedInferenceServerTrain sharedSrv(backend);

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());

    // Tune lane is lighter than arena: only 2 GameContext, one shared NN server for all.
    const unsigned threadsPerSide = 1;
    const unsigned wantedLanes = (unsigned)std::max(1, (TOTAL_GAMES + 1) / 2);
    const unsigned lanesByFormula = (hw > 4u) ? ((hw - 4u) / 2u) : 1u;
    const unsigned parallelLanes = std::max(1u, std::min(wantedLanes, lanesByFormula));

    sharedSrv.start();

    std::vector<std::unique_ptr<TuneLane>> lanes;
    lanes.reserve(parallelLanes);
    LanesStopGuard<TuneLane> lanesGuard(lanes);

    struct TuneCleanupGuard {
        SharedInferenceServerTrain* srv = nullptr;
        ~TuneCleanupGuard() noexcept {
            try {
                if (srv) {
                    srv->requestStop();
                    srv->join();
                }
            }
            catch (...) {}
        }
    } srvGuard{ &sharedSrv };

    for (unsigned i = 0; i < parallelLanes; ++i) {
        auto lane = std::make_unique<TuneLane>();
        lane->start(threadsPerSide);
        lanes.push_back(std::move(lane));
    }

    std::cout
        << "[tune] start\n"
        << "  P1: c_init=" << c_init1 << " fpu_reduction=" << fpu_reduction1 << "\n"
        << "  P2: c_init=" << c_init2 << " fpu_reduction=" << fpu_reduction2 << "\n"
        << "  games=" << TOTAL_GAMES << " sims=" << SIMS_PER_POS << "\n"
        << "  parallel_lanes=" << parallelLanes
        << " threads_per_side=" << threadsPerSide << "\n";

    auto onProgress = [&](int playedGames, const MatchStatsGeneric& s) {
        if ((playedGames % 2) == 0 && (playedGames % 100) == 0) {
            printTuneProgress(playedGames, s.p1Wins, s.p2Wins);
        }
        };

    MatchStatsGeneric g = runUniversalMatchEngine(
        lanes,
        TOTAL_GAMES,
        [&](TuneLane& lane,
            const Position& startPos,
            const std::array<uint64_t, 4>& path,
            const std::array<int, 64>& mask,
            bool p1IsWhite,
            const std::vector<int>* mirroredDice,
            std::vector<int>* producedDice) -> int {
                return playOneTuneGameOnLane(
                    lane,
                    sharedSrv,
                    backend,
                    p1,
                    p2,
                    startPos,
                    path,
                    mask,
                    p1IsWhite,
                    SIMS_PER_POS,
                    MAX_PLIES,
                    mirroredDice,
                    producedDice
                );
        },
        onProgress,
        /*progressEveryPairs=*/50);

    printTuneProgress(TOTAL_GAMES, g.p1Wins, g.p2Wins);
    std::cout << "[tune] finished\n";
}
static float lambdaQ = 1;//ok
static float lambdaD = 1;//ok
static float lambdaC = 0.8;//ok
static float lambdaT = 1;//ok
static float lambdaS = 0;//ok
static float lambdaZ = 0;//ok
static AI_FORCEINLINE float valueToSidePerspective(float v, int fromSide, int toSide) {
    v = clamp01(v);
    return (fromSide == toSide) ? v : (1.0f - v);
}

static AI_FORCEINLINE float chanceStepDecay(uint8_t chanceCount) {
    if (chanceCount)return lambdaC;
    return lambdaD;
}

static void buildChanceWeightedTargets(
    std::vector<TrainSample>& game,
    const std::vector<int>& sideAtSample,
    const std::vector<uint8_t>& chanceToNext,
    float zWhite)
{
    const int n = (int)game.size();
    if (n <= 0) return;
    if ((int)sideAtSample.size() != n || (int)chanceToNext.size() != n) return;

    for (int i = 0; i < n; ++i) {
        const int sideCur = sideAtSample[(size_t)i];
        float v = lambdaQ;
        float sumV = v;
        float weighted = v * clamp01(game[(size_t)i].q);

        for (int j = i + 1; j < n; ++j) {
            v *= chanceStepDecay(chanceToNext[(size_t)j - 1]);
            sumV += v;

            const int sideJ = sideAtSample[(size_t)j];
            const float qInCurPerspective =
                valueToSidePerspective(game[(size_t)j].q, sideJ, sideCur);
            weighted += v * qInCurPerspective;
        }

        v = v * chanceStepDecay(chanceToNext[(size_t)n - 1]) * lambdaT + sumV * lambdaS + lambdaZ;
        sumV += v;

        const float zCur = (sideCur == 0) ? zWhite : (1.0f - zWhite);
        weighted += v * clamp01(zCur);

        game[(size_t)i].z = clamp01(weighted / std::max(sumV, 1e-12f));
    }
}

static void selfPlayOneGame960(GameContext& sp,
    ITrainInferenceServer& sharedSrv,
    BackendBinding backend,
    ReplayBuffer& rb,
    int simsPerPos,
    int maxPlies,
    bool addRootNoise,
    int& outPlyCount,
    bool& outTerminated,
    int& outSamplesAdded) {
    sp.resetForNewGame();

    Position pos;
    array<uint64_t, 4> path;
    array<int, 64> mask;

    std::vector<TrainSample> game;
    std::vector<int> sideAtSample;
    std::vector<uint8_t> chanceToNext;

    MoveList ml;
    int term = 0;

    std::vector<moveState> moves;

    TrainSample sample;
    int d = 0;

    chess960(pos, path, mask);

    game.reserve((size_t)maxPlies);
    sideAtSample.reserve((size_t)maxPlies);
    chanceToNext.reserve((size_t)maxPlies);

    outTerminated = false;
    outSamplesAdded = 0;

    for (int ply = 0; ply < maxPlies; ++ply) {
        // Early stop if table overflow
        if (sp.T.abort.load(std::memory_order_relaxed)) break;

        genLegal(pos, path, mask, ml, term);

        if (term) { outTerminated = true; break; }

        if (ml.n == 0) {
            if (!chanceToNext.empty() && chanceToNext.back() < 255u) {
                ++chanceToNext.back();
            }
            makeRandom(pos, sp.T.findNodeNoInsert(pos.key));
            continue;
        }

        bool rootNoiseHere = addRootNoise && (d < 20);

        runFixedSims(sp.T, sp.pool, sharedSrv, backend,
            pos, path, mask, simsPerPos, rootNoiseHere);
        if (sp.T.abort.load(std::memory_order_relaxed)) break;

        collectRootMoves(sp.T, pos, sample.q, moves);

        if (moves.empty()) break;

        sample.pos = pos;
        buildSparsePolicyTargetCHW(pos, moves, sample.nPi, sample.piIdx, sample.piProbQ);

        game.push_back(sample);
        sideAtSample.push_back(pos.side);
        chanceToNext.push_back(0);

        float temp = (d < 20) ? 1.0f : 0.0f;
        int mv = pickMoveFromVisits(moves, temp);
        if (!mv) break;

        makeMove(pos, mask, mv);
        ++d;
    }

    outPlyCount = d;

    float zWhite = 0.5f;
    if (outTerminated) {
        // winner = side-to-move => whiteWin = 1 - pos.side
        zWhite = 1.0f - pos.side;
    }
    else return;

    // Weighted q target accounting for number of chance transitions between samples.
    buildChanceWeightedTargets(game, sideAtSample, chanceToNext, zWhite);

    rb.pushMany(game);
    outSamplesAdded += (int)game.size();
}

// ------------------------------------------------------------
// Trainer thread: sparse policy loss via gather(logp, idx)
// + pin_memory/non_blocking, + grad clipping, + NaN guard
// ------------------------------------------------------------

struct TrainerState {
    std::atomic<bool> stop{ false };
    std::atomic<uint64_t> steps{ 0 };
    std::atomic<float> lastLoss{ 0.0f };
};
struct TensorPairRef {
    torch::Tensor dst;
    torch::Tensor src;
};

struct ModulePairCache {
    std::vector<TensorPairRef> params;
    std::vector<TensorPairRef> buffers;

    void clear() {
        params.clear();
        buffers.clear();
    }

    bool empty() const {
        return params.empty() && buffers.empty();
    }
};

static ModulePairCache buildModulePairCache(Net& dst, Net& src,
    const char* tag = "ModulePairCache") {
    ModulePairCache cache;

    auto srcParams = src->named_parameters(true);
    auto dstParams = dst->named_parameters(true);

    cache.params.reserve(srcParams.size());
    for (const auto& kv : srcParams) {
        auto* d = dstParams.find(kv.key());
        if (!d) {
            std::ostringstream oss;
            oss << "[" << tag << "] missing dst parameter: " << kv.key();
            throw std::runtime_error(oss.str());
        }
        cache.params.push_back(TensorPairRef{ *d, kv.value() });
    }

    auto srcBufs = src->named_buffers(true);
    auto dstBufs = dst->named_buffers(true);

    cache.buffers.reserve(srcBufs.size());
    for (const auto& kv : srcBufs) {
        auto* d = dstBufs.find(kv.key());
        if (!d) {
            std::ostringstream oss;
            oss << "[" << tag << "] missing dst buffer: " << kv.key();
            throw std::runtime_error(oss.str());
        }
        cache.buffers.push_back(TensorPairRef{ *d, kv.value() });
    }

    return cache;
}

static void emaUpdateCached(ModulePairCache& cache, double decay) {
    torch::NoGradGuard ng;

    for (auto& p : cache.params) {
        auto s = p.src.detach().to(
            p.dst.device(),
            p.dst.scalar_type(),
            /*non_blocking=*/false,
            /*copy=*/false
        );

        p.dst.mul_(decay);
        p.dst.add_(s, 1.0 - decay);
    }

    for (auto& b : cache.buffers) {
        b.dst.copy_(b.src.detach().to(
            b.dst.device(),
            b.dst.scalar_type(),
            /*non_blocking=*/false,
            /*copy=*/false
        ));
    }
}

static void copyNetStateCached(ModulePairCache& cache) {
    torch::NoGradGuard ng;

    for (auto& p : cache.params) {
        p.dst.copy_(p.src.detach().to(
            p.dst.device(),
            p.dst.scalar_type(),
            /*non_blocking=*/false,
            /*copy=*/false
        ));
    }

    for (auto& b : cache.buffers) {
        b.dst.copy_(b.src.detach().to(
            b.dst.device(),
            b.dst.scalar_type(),
            /*non_blocking=*/false,
            /*copy=*/false
        ));
    }
}
struct CudaAutocastGuard {
    bool enabled = false;
    bool prevEnabled = false;
    bool prevCacheEnabled = false;
    at::ScalarType prevDtype = at::kFloat;

    explicit CudaAutocastGuard(bool en,
        at::ScalarType dtype = at::kHalf,
        bool cacheEnabled = true)
        : enabled(en) {
        if (!enabled) return;

        prevEnabled = at::autocast::is_autocast_enabled(at::kCUDA);
        prevCacheEnabled = at::autocast::is_autocast_cache_enabled();
        prevDtype = at::autocast::get_autocast_dtype(at::kCUDA);

        at::autocast::increment_nesting();
        at::autocast::set_autocast_enabled(at::kCUDA, true);
        at::autocast::set_autocast_dtype(at::kCUDA, dtype);
        at::autocast::set_autocast_cache_enabled(cacheEnabled);
    }

    ~CudaAutocastGuard() {
        if (!enabled) return;

        at::autocast::set_autocast_enabled(at::kCUDA, prevEnabled);
        at::autocast::set_autocast_dtype(at::kCUDA, prevDtype);
        at::autocast::set_autocast_cache_enabled(prevCacheEnabled);

        if (at::autocast::decrement_nesting() == 0) {
            at::autocast::clear_cache();
        }
    }
};

struct SimpleGradScaler {
    bool enabled = false;

    float scale = 65536.0f;
    float growthFactor = 2.0f;
    float backoffFactor = 0.5f;
    int growthInterval = 2000;
    int growthTracker = 0;
    float minScale = 1.0f;

    torch::Tensor scaleLoss(const torch::Tensor& loss) const {
        if (!enabled) return loss;
        return loss * scale;
    }

    void unscale(const std::vector<torch::Tensor>& params) {
        if (!enabled) return;

        const float invScale = 1.0f / scale;
        for (const auto& p : params) {
            auto g = p.grad();
            if (!g.defined()) continue;
            g.mul_(invScale);
        }
    }

    void update(bool gradsFinite) {
        if (!enabled) return;

        if (!gradsFinite) {
            scale = std::max(minScale, scale * backoffFactor);
            growthTracker = 0;
            return;
        }

        ++growthTracker;
        if (growthTracker >= growthInterval) {
            scale *= growthFactor;
            growthTracker = 0;
        }
    }
};

struct TrainTensorStage {
    torch::Tensor x;
    torch::Tensor idx;
    torch::Tensor prob;
    torch::Tensor z;
    torch::Tensor nPi;
};

struct Trainer {
    torch::Device device{ torch::kCPU };
    bool useCuda = false;

    bool useAmp = false;
    at::ScalarType ampDtype = at::kHalf;
    SimpleGradScaler scaler;

    uint64_t ampSkippedSteps = 0;
    float lastAmpScale = 1.0f;

    // Hyperparams
    double initial_lr = 1e-4;
    double min_lr = 1e-4;
    double current_lr = initial_lr;
    double wd = 1e-4;
    double ema_decay = 0.999;

    // Cosine Annealing with Warmup
    uint64_t lr_warmup_steps = 10000;
    uint64_t lr_total_steps = 1000000;
    double   lr_warmup_start_factor = 0.10;

    // Optional short smoothing after process restart/resume
    uint64_t resumeStartStep = 0;
    uint64_t warmupStepsAfterRestart = 2000;
    double   warmupStartFactor = 0.10;

    // Batch
    int B = 256;

    // RNG
    std::mt19937 rng{ 0xBADC0FFEu };

    // Optimizer
    std::unique_ptr<torch::optim::AdamW> opt;

    // Cached EMA dst/src tensor pairs
    ModulePairCache emaCache;

    // Double-buffered CPU staging (pinned if CUDA)
    std::array<TrainTensorStage, 2> hostStage{};

    // Double-buffered device tensors
    std::array<TrainTensorStage, 2> devStage{};

    // [1, AI_MAX_MOVES] => 0..254 on the active device
    torch::Tensor slotIdsDev;

    // Async H2D pipeline
    cudaStream_t h2dStream = nullptr;
    std::array<cudaEvent_t, 2> h2dDone{ {nullptr, nullptr} };
    std::array<cudaEvent_t, 2> computeDone{ {nullptr, nullptr} };
    std::array<bool, 2> h2dDoneValid{ {false, false} };
    std::array<bool, 2> computeDoneValid{ {false, false} };

    // State
    uint64_t steps = 0;
    float lastLoss = 0.0f;
    float lastLossP = 0.0f;
    float lastLossV = 0.0f;
    float lastEntropy = 0.0f;
    float lastVMAE = 0.0f;
    float lastGradNorm = 0.0f;

    ~Trainer() {
        shutdownAsyncPipeline();
    }

    static AI_FORCEINLINE size_t tensorBytes(const torch::Tensor& t) {
        return (size_t)t.numel() * (size_t)t.element_size();
    }

    AI_FORCEINLINE cudaStream_t currentComputeStream() const {
        if (!useCuda) return nullptr;
        return at::cuda::getCurrentCUDAStream(device.index()).stream();
    }

    void initAsyncPipeline() {
        if (!useCuda) return;

        if (!h2dStream) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&h2dStream, cudaStreamNonBlocking));
        }

        for (int s = 0; s < 2; ++s) {
            if (!h2dDone[(size_t)s]) {
                CUDA_CHECK(cudaEventCreateWithFlags(&h2dDone[(size_t)s], cudaEventDisableTiming));
            }
            if (!computeDone[(size_t)s]) {
                CUDA_CHECK(cudaEventCreateWithFlags(&computeDone[(size_t)s], cudaEventDisableTiming));
            }
        }

        cudaStream_t cs = currentComputeStream();
        for (int s = 0; s < 2; ++s) {
            CUDA_CHECK(cudaEventRecord(h2dDone[(size_t)s], cs));
            CUDA_CHECK(cudaEventRecord(computeDone[(size_t)s], cs));
            h2dDoneValid[(size_t)s] = true;
            computeDoneValid[(size_t)s] = true;
        }
    }

    void shutdownAsyncPipeline() {
        if (!useCuda) return;

        if (h2dStream) {
            cudaStreamSynchronize(h2dStream);
        }

        for (int s = 0; s < 2; ++s) {
            if (h2dDone[(size_t)s]) {
                cudaEventDestroy(h2dDone[(size_t)s]);
                h2dDone[(size_t)s] = nullptr;
            }
            if (computeDone[(size_t)s]) {
                cudaEventDestroy(computeDone[(size_t)s]);
                computeDone[(size_t)s] = nullptr;
            }
            h2dDoneValid[(size_t)s] = false;
            computeDoneValid[(size_t)s] = false;
        }

        if (h2dStream) {
            cudaStreamDestroy(h2dStream);
            h2dStream = nullptr;
        }
    }

    void waitHostSlotReusable(int slot) {
        if (!useCuda) return;
        if (!h2dDoneValid[(size_t)slot]) return;
        CUDA_CHECK(cudaEventSynchronize(h2dDone[(size_t)slot]));
    }

    void enqueueStageToDeviceAsync(int slot) {
        if (!useCuda) return;

        if (computeDoneValid[(size_t)slot]) {
            CUDA_CHECK(cudaStreamWaitEvent(h2dStream, computeDone[(size_t)slot], 0));
        }

        CUDA_CHECK(cudaMemcpyAsync(devStage[(size_t)slot].x.data_ptr<float>(), hostStage[(size_t)slot].x.data_ptr<float>(), tensorBytes(hostStage[(size_t)slot].x), cudaMemcpyHostToDevice, h2dStream));
        CUDA_CHECK(cudaMemcpyAsync(devStage[(size_t)slot].idx.data_ptr<int64_t>(), hostStage[(size_t)slot].idx.data_ptr<int64_t>(), tensorBytes(hostStage[(size_t)slot].idx), cudaMemcpyHostToDevice, h2dStream));
        CUDA_CHECK(cudaMemcpyAsync(devStage[(size_t)slot].prob.data_ptr<float>(), hostStage[(size_t)slot].prob.data_ptr<float>(), tensorBytes(hostStage[(size_t)slot].prob), cudaMemcpyHostToDevice, h2dStream));
        CUDA_CHECK(cudaMemcpyAsync(devStage[(size_t)slot].z.data_ptr<float>(), hostStage[(size_t)slot].z.data_ptr<float>(), tensorBytes(hostStage[(size_t)slot].z), cudaMemcpyHostToDevice, h2dStream));
        CUDA_CHECK(cudaMemcpyAsync(devStage[(size_t)slot].nPi.data_ptr<int64_t>(), hostStage[(size_t)slot].nPi.data_ptr<int64_t>(), tensorBytes(hostStage[(size_t)slot].nPi), cudaMemcpyHostToDevice, h2dStream));

        CUDA_CHECK(cudaEventRecord(h2dDone[(size_t)slot], h2dStream));
        h2dDoneValid[(size_t)slot] = true;
    }

    void waitDeviceSlotReadyOnComputeStream(int slot) {
        if (!useCuda) return;
        if (!h2dDoneValid[(size_t)slot]) return;
        CUDA_CHECK(cudaStreamWaitEvent(currentComputeStream(), h2dDone[(size_t)slot], 0));
    }

    void markComputeUsesSlot(int slot) {
        if (!useCuda) return;
        CUDA_CHECK(cudaEventRecord(computeDone[(size_t)slot], currentComputeStream()));
        computeDoneValid[(size_t)slot] = true;
    }

    static constexpr double kPi = 3.1415926535897932384626433832795;

    double computeCosineBaseLR(uint64_t s) const {
        const double base = initial_lr;
        const double floor = std::min(min_lr, base);

        if (lr_total_steps <= 1) return floor;

        if (s < lr_warmup_steps) {
            const double t = (double)s / (double)std::max<uint64_t>(1, lr_warmup_steps);
            const double mult = lr_warmup_start_factor + (1.0 - lr_warmup_start_factor) * t;
            return base * mult;
        }

        const uint64_t decaySpan = (lr_total_steps > lr_warmup_steps) ? (lr_total_steps - lr_warmup_steps) : 1ull;
        const uint64_t decayPos = std::min<uint64_t>(s - lr_warmup_steps, decaySpan);

        const double u = (double)decayPos / (double)decaySpan;
        const double c = 0.5 * (1.0 + std::cos(kPi * u));

        return floor + (base - floor) * c;
    }

    double computeRestartWarmupMultiplier(uint64_t s) const {
        if (warmupStepsAfterRestart == 0) return 1.0;

        uint64_t delta = (s >= resumeStartStep) ? (s - resumeStartStep) : 0;
        if (delta >= warmupStepsAfterRestart) return 1.0;

        double t = (double)delta / (double)warmupStepsAfterRestart;
        return warmupStartFactor + (1.0 - warmupStartFactor) * t;
    }

    void updateLR(bool forceLog = false) {
        const double prev = current_lr;

        const double cosine_lr = computeCosineBaseLR(steps);
        const double restart_mult = computeRestartWarmupMultiplier(steps);
        const double target_lr = cosine_lr * restart_mult;

        for (auto& group : opt->param_groups()) {
            static_cast<torch::optim::AdamWOptions&>(group.options()).lr(target_lr);
        }

        current_lr = target_lr;

        const bool changed = std::fabs(current_lr - prev) > 1e-15;
        const bool restartWarmupJustFinished =
            (warmupStepsAfterRestart > 0 &&
                steps == resumeStartStep + warmupStepsAfterRestart);

        (void)forceLog;
        (void)changed;
        (void)restartWarmupJustFinished;
    }
    static AI_FORCEINLINE bool endsWithStr(const std::string& s, const char* suf) {
        const size_t n = std::strlen(suf);
        return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
    }

    static AI_FORCEINLINE void fillStageFromBatch(
        TrainTensorStage& st,
        const std::vector<TrainSample>& batch,
        int B)
    {
        float* xp = st.x.data_ptr<float>();
        int64_t* ip = st.idx.data_ptr<int64_t>();
        float* pp = st.prob.data_ptr<float>();
        float* zp = st.z.data_ptr<float>();
        int64_t* np = st.nPi.data_ptr<int64_t>();

        for (int i = 0; i < B; ++i) {
            const TrainSample& s = batch[(size_t)i];

            NNInput enc;
            positionToNNInput(s.pos, enc);

            std::memcpy(
                xp + (size_t)i * (size_t)NN_INPUT_SIZE,
                enc.data(),
                (size_t)NN_INPUT_SIZE * sizeof(float)
            );

            np[(size_t)i] = (int64_t)s.nPi;

            decodeTrainSamplePolicyRow(
                s,
                ip + (size_t)i * (size_t)AI_MAX_MOVES,
                pp + (size_t)i * (size_t)AI_MAX_MOVES
            );

            zp[(size_t)i] = s.z;
        }
    }

    void copyStageToDevice(int slot) {
        if (!useCuda) return;
        enqueueStageToDeviceAsync(slot);
    }
    void init(Net& model, Net& emaModel) {
        try { torch::set_num_threads(1); }
        catch (...) {}
        try { torch::set_num_interop_threads(1); }
        catch (...) {}

        device = torch::Device(torch::kCPU);
        useCuda = false;

        try {
            if (torch::cuda::is_available() && torch::cuda::device_count() > 0) {
                device = torch::Device(torch::kCUDA, 0);
                useCuda = true;
            }
        }
        catch (...) {
            device = torch::Device(torch::kCPU);
            useCuda = false;
        }

        useAmp = useCuda;          // AMP only on CUDA
        ampDtype = at::kHalf;      // fp16 autocast on CUDA

        scaler.enabled = useAmp;
        scaler.scale = 65536.0f;
        scaler.growthFactor = 2.0f;
        scaler.backoffFactor = 0.5f;
        scaler.growthInterval = 2000;
        scaler.growthTracker = 0;
        scaler.minScale = 1.0f;
        lastAmpScale = scaler.scale;


        {
            std::lock_guard<std::mutex> lk(g_modelMutex);
            model->to(device);
            model->train();

            emaModel->to(device);
            emaModel->eval();

            // Build EMA cache AFTER final device placement.
            emaCache = buildModulePairCache(emaModel, model, "emaCache");
        }

        // AdamW with no weight decay on BN / bias
        std::vector<torch::Tensor> decayParams;
        std::vector<torch::Tensor> noDecayParams;

        {
            auto named = model->named_parameters(/*recurse=*/true);
            for (const auto& kv : named) {
                const std::string name = kv.key();
                const torch::Tensor& p = kv.value();

                if (!p.defined() || !p.requires_grad()) continue;

                const bool isBias = endsWithStr(name, ".bias");
                const bool is1D = p.dim() <= 1;

                if (isBias || is1D) noDecayParams.push_back(p);
                else                decayParams.push_back(p);
            }
        }

        const size_t decayCount = decayParams.size();
        const size_t noDecayCount = noDecayParams.size();

        if (decayParams.empty() && noDecayParams.empty()) {
            throw std::runtime_error("Trainer::init(): model has no trainable parameters");
        }

        // Base optimizer group:
        // prefer decayParams as the main group; if empty, bootstrap from noDecayParams.
        if (!decayParams.empty()) {
            opt = std::make_unique<torch::optim::AdamW>(
                decayParams,
                torch::optim::AdamWOptions(initial_lr).weight_decay(wd)
            );
        }
        else {
            opt = std::make_unique<torch::optim::AdamW>(
                noDecayParams,
                torch::optim::AdamWOptions(initial_lr).weight_decay(0.0)
            );
            noDecayParams.clear(); // already used as base group
        }

        // Add no-decay group only if it wasn't already consumed above.
        if (!noDecayParams.empty()) {
            auto opts = torch::optim::AdamWOptions(initial_lr);
            opts.weight_decay(0.0);

            torch::optim::OptimizerParamGroup g(
                noDecayParams,
                std::make_unique<torch::optim::AdamWOptions>(opts)
            );

            opt->add_param_group(std::move(g));
        }

        (void)decayCount;
        (void)noDecayCount;

        resumeStartStep = steps;
        current_lr = -1.0;
        updateLR(true);

        auto makeCPU = [&](torch::IntArrayRef sizes, torch::ScalarType t) {
            auto ten = torch::empty(sizes, torch::TensorOptions().dtype(t).device(torch::kCPU));
            if (useCuda) ten = ten.pin_memory();
            return ten;
            };

        for (int s = 0; s < 2; ++s) {
            hostStage[(size_t)s].x = makeCPU({ B, NN_SQ_PLANES, 8, 8 }, torch::kFloat32);
            hostStage[(size_t)s].idx = makeCPU({ B, AI_MAX_MOVES }, torch::kInt64);
            hostStage[(size_t)s].prob = makeCPU({ B, AI_MAX_MOVES }, torch::kFloat32);
            hostStage[(size_t)s].z = makeCPU({ B, 1 }, torch::kFloat32);
            hostStage[(size_t)s].nPi = makeCPU({ B }, torch::kInt64);
        }

        if (useCuda) {
            for (int s = 0; s < 2; ++s) {
                devStage[(size_t)s].x = torch::empty(
                    { B, NN_SQ_PLANES, 8, 8 },
                    torch::TensorOptions().dtype(torch::kFloat32).device(device));

                devStage[(size_t)s].idx = torch::empty(
                    { B, AI_MAX_MOVES },
                    torch::TensorOptions().dtype(torch::kInt64).device(device));

                devStage[(size_t)s].prob = torch::empty(
                    { B, AI_MAX_MOVES },
                    torch::TensorOptions().dtype(torch::kFloat32).device(device));

                devStage[(size_t)s].z = torch::empty(
                    { B, 1 },
                    torch::TensorOptions().dtype(torch::kFloat32).device(device));

                devStage[(size_t)s].nPi = torch::empty(
                    { B },
                    torch::TensorOptions().dtype(torch::kInt64).device(device));
            }
        }
        else {
            // CPU fallback: just alias host buffers
            for (int s = 0; s < 2; ++s) {
                devStage[(size_t)s] = hostStage[(size_t)s];
            }
        }

        // slot ids for masking legal part: [1, 255] = 0..254
        slotIdsDev = torch::arange(
            AI_MAX_MOVES,
            torch::TensorOptions().dtype(torch::kInt64).device(device)
        ).view({ 1, AI_MAX_MOVES });

        if (useCuda) {
            initAsyncPipeline();
        }
    }

    int trainBlockBudgetMs(ReplayBuffer& rb, Net& model, Net& emaModel,
        int budgetMs,
        int maxStepsHard,
        int warmupBatches = 1000) {
        if (budgetMs <= 0 || maxStepsHard <= 0) return 0;

        const size_t need = (size_t)B * (size_t)std::max(1, warmupBatches);
        if (rb.currentSize() < need) return 0;

        const auto tEnd = std::chrono::steady_clock::now() + std::chrono::milliseconds(budgetMs);

        std::array<std::vector<TrainSample>, 2> batchBuf;
        batchBuf[0].reserve((size_t)B);
        batchBuf[1].reserve((size_t)B);

        int done = 0;

        int skippedConsecutive = 0;
        int skippedTotal = 0;
        static constexpr int MAX_SKIPPED_CONSECUTIVE = 32;
        static constexpr int MAX_SKIPPED_TOTAL = 256;

        // Trainer::trainBlockBudgetMs
        static constexpr uint64_t HOST_STATS_EVERY = 64;

        int cur = 0;
        int next = 1;

        // Preload first batch
        if (!rb.sampleBatch(batchBuf[(size_t)cur], B, rng)) return 0;

        if (useCuda) {
            waitHostSlotReusable(cur);
        }
        fillStageFromBatch(hostStage[(size_t)cur], batchBuf[(size_t)cur], B);
        if (useCuda) {
            enqueueStageToDeviceAsync(cur);
        }

        for (int it = 0; it < maxStepsHard; ++it) {
            if (std::chrono::steady_clock::now() >= tEnd) break;

            torch::Tensor xBatch;
            torch::Tensor idxBatch;
            torch::Tensor probBatch;
            torch::Tensor zBatch;
            torch::Tensor nPiBatch;

            if (useCuda) {
                waitDeviceSlotReadyOnComputeStream(cur);

                xBatch = devStage[(size_t)cur].x;
                idxBatch = devStage[(size_t)cur].idx;
                probBatch = devStage[(size_t)cur].prob;
                zBatch = devStage[(size_t)cur].z;
                nPiBatch = devStage[(size_t)cur].nPi;
            }
            else {
                xBatch = hostStage[(size_t)cur].x;
                idxBatch = hostStage[(size_t)cur].idx;
                probBatch = hostStage[(size_t)cur].prob;
                zBatch = hostStage[(size_t)cur].z;
                nPiBatch = hostStage[(size_t)cur].nPi;
            }

            const bool needHostStats =
                (((steps + (uint64_t)done + 1ull) % HOST_STATS_EVERY) == 0ull);

            float lossScalar = lastLoss;
            float lossPScalar = lastLossP;
            float lossVScalar = lastLossV;
            float entropyScalar = lastEntropy;
            float vMaeScalar = lastVMAE;
            float gradNormScalar = lastGradNorm;
            bool didStep = false;

            {
                std::lock_guard<std::mutex> lk(g_modelMutex);

                opt->zero_grad();

                auto runForwardLoss = [&]() {
                    auto out = model->forward(xBatch);
                    auto pol = out.first;         // [B,73,8,8]
                    auto valLogits = out.second;  // [B,1]

                    // =========================================================
                    // POLICY LOSS over LEGAL MOVES ONLY
                    // =========================================================
                    auto polFlat = pol.flatten(1).to(torch::kFloat32); // [B, POLICY_SIZE]

                    auto nPiClamped = nPiBatch.clamp(0, AI_MAX_MOVES); // [B], int64
                    auto validMask = slotIdsDev.lt(nPiClamped.view({ -1, 1 })); // [B, AI_MAX_MOVES], bool

                    auto idxSafe = idxBatch.clamp(0, POLICY_SIZE - 1); // [B, AI_MAX_MOVES], int64
                    auto gathered = polFlat.gather(1, idxSafe);        // [B, AI_MAX_MOVES], FP32

                    constexpr float kMaskedLogit = -1e9f;
                    auto maskedLogits = gathered.masked_fill(torch::logical_not(validMask), kMaskedLogit);

                    auto logp_valid = torch::log_softmax(maskedLogits, 1); // [B, AI_MAX_MOVES], FP32
                    auto p_valid = torch::softmax(maskedLogits, 1);        // [B, AI_MAX_MOVES], FP32

                    auto tgtProb = probBatch.to(torch::kFloat32)
                        .masked_fill(torch::logical_not(validMask), 0.0f);

                    auto rowLossP = -(tgtProb * logp_valid).sum(1); // [B]

                    auto rowHasTarget = nPiClamped.gt(0).to(torch::kFloat32); // [B]
                    auto denomP = rowHasTarget.sum().clamp_min(1.0f);

                    auto lossP = (rowLossP * rowHasTarget).sum() / denomP;

                    // Entropy model policy on legal moves
                    auto rowEntropy = -(p_valid * logp_valid).sum(1); // [B]
                    auto entropy = (rowEntropy * rowHasTarget).sum() / denomP;

                    // =========================================================
                    // VALUE LOSS
                    // =========================================================
                    auto zF = zBatch.to(torch::kFloat32);
                    auto valLogitsF = valLogits.to(torch::kFloat32);

                    auto lossV = torch::binary_cross_entropy_with_logits(valLogitsF, zF);

                    auto valProb = torch::sigmoid(valLogitsF);
                    auto vMAE = torch::mean(torch::abs(valProb - zF));

                    auto loss = lossP + lossV;
                    return std::make_tuple(loss, lossP, lossV, entropy, vMAE);
                    };

                torch::Tensor loss, lossP, lossV, entropyT, vMAET;

                if (useAmp) {
                    CudaAutocastGuard ampGuard(true, ampDtype);
                    std::tie(loss, lossP, lossV, entropyT, vMAET) = runForwardLoss();
                }
                else {
                    std::tie(loss, lossP, lossV, entropyT, vMAET) = runForwardLoss();
                }

                const bool finiteLoss = torch::isfinite(loss).all().item<bool>();
                if (finiteLoss) {
                    if (useAmp) {
                        scaler.scaleLoss(loss).backward();

                        scaler.unscale(model->parameters());

                        double currentGradNorm =
                            torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);

                        bool gradsFinite = std::isfinite(currentGradNorm);

                        if (gradsFinite) {
                            opt->step();
                            emaUpdateCached(emaCache, ema_decay);

                            if (needHostStats) {
                                lossScalar = loss.detach().item<float>();
                                lossPScalar = lossP.detach().item<float>();
                                lossVScalar = lossV.detach().item<float>();
                                entropyScalar = entropyT.detach().item<float>();
                                vMaeScalar = vMAET.detach().item<float>();
                            }

                            gradNormScalar = static_cast<float>(currentGradNorm);
                            didStep = true;
                        }

                        scaler.update(gradsFinite);
                        lastAmpScale = scaler.scale;

                        if (!didStep) {
                            ++ampSkippedSteps;
                        }
                    }
                    else {
                        loss.backward();

                        double currentGradNorm =
                            torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);

                        if (std::isfinite(currentGradNorm)) {
                            opt->step();
                            emaUpdateCached(emaCache, ema_decay);

                            if (needHostStats) {
                                lossScalar = loss.detach().item<float>();
                                lossPScalar = lossP.detach().item<float>();
                                lossVScalar = lossV.detach().item<float>();
                                entropyScalar = entropyT.detach().item<float>();
                                vMaeScalar = vMAET.detach().item<float>();
                            }

                            gradNormScalar = static_cast<float>(currentGradNorm);
                            didStep = true;
                        }
                    }
                }
                else {
                    if (useAmp) {
                        scaler.update(false);
                        lastAmpScale = scaler.scale;
                        ++ampSkippedSteps;
                    }
                }
            }

            if (!didStep) {
                ++skippedConsecutive;
                ++skippedTotal;

                (void)lastAmpScale;

                if (skippedConsecutive >= MAX_SKIPPED_CONSECUTIVE ||
                    skippedTotal >= MAX_SKIPPED_TOTAL) {
                    break;
                }
            }
            else {
                skippedConsecutive = 0;

                ++done;
                ++steps;

                if (needHostStats) {
                    lastLoss = lossScalar;
                    lastLossP = lossPScalar;
                    lastLossV = lossVScalar;
                    lastEntropy = entropyScalar;
                    lastVMAE = vMaeScalar;
                }

                lastGradNorm = gradNormScalar;
                updateLR();
            }

            if (useCuda) {
                markComputeUsesSlot(cur);
            }

            // ---------------------------------------------------------
            // Prefetch/build next batch on CPU while current GPU work is
            // still draining asynchronously.
            // ---------------------------------------------------------
            if ((it + 1) >= maxStepsHard) break;
            if (std::chrono::steady_clock::now() >= tEnd) break;

            if (!rb.sampleBatch(batchBuf[(size_t)next], B, rng)) break;

            if (useCuda) {
                waitHostSlotReusable(next);
            }
            fillStageFromBatch(hostStage[(size_t)next], batchBuf[(size_t)next], B);
            if (useCuda) {
                enqueueStageToDeviceAsync(next);
            }

            std::swap(cur, next);
        }

        if (useCuda) {
            try { torch::cuda::synchronize(); }
            catch (...) {}
        }

        return done;
    }
};
// init / load / save
// ------------------------------------------------------------
// ------------------------------------------------------------
// Trainer checkpoint: optimizer state
// ------------------------------------------------------------

static bool saveOptimizerState(const std::string& optFile, const Trainer& trainer) {
    if (!trainer.opt) return false;

    try {
        torch::serialize::OutputArchive ar;
        trainer.opt->save(ar);
        ar.save_to(optFile);
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "saveOptimizerState failed: " << e.what() << "\n";
        return false;
    }
}

static bool loadOptimizerState(const std::string& optFile, Trainer& trainer) {
    if (!trainer.opt) return false;
    if (!fileExists(optFile)) return false;

    try {
        torch::serialize::InputArchive ar;
        ar.load_from(optFile);
        trainer.opt->load(ar);
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "loadOptimizerState failed: " << e.what() << "\n";
        return false;
    }
}

static bool loadOrCreateTorchModel(const std::string& ptFile, Net& model) {
    if (fileExists(ptFile)) {
        try {
            torch::load(model, ptFile);
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "torch::load failed: " << e.what() << "\n";
            return false;
        }
    }

    try {
        torch::save(model, ptFile);
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "torch::save (create) failed: " << e.what() << "\n";
        return false;
    }
}

static void copyNetState(Net& dst, Net& src) {
    auto cache = buildModulePairCache(dst, src, "copyNetState");
    copyNetStateCached(cache);

    if (src->is_training()) dst->train();
    else                    dst->eval();
}

static bool loadOrCreateEmaModel(const std::string& emaFile, Net& emaModel, Net& model) {
    if (fileExists(emaFile)) {
        try {
            torch::load(emaModel, emaFile);
            emaModel->eval();
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "torch::load(ema) failed: " << e.what()
                << " -> fallback to current model\n";
        }
    }

    try {
        copyNetState(emaModel, model);
        emaModel->eval();
        torch::save(emaModel, emaFile);
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "create/save ema model failed: " << e.what() << "\n";
        return false;
    }
}



static bool ensureOldRunnerReady(const std::string& planFile) {
    std::lock_guard<std::mutex> lk(g_trtOldMutex);
    if (g_trtOldReady) return true;

    if (!g_trt_old.initOrCreate(planFile)) return false;
    g_trtOldReady = true;
    return true;
}

static bool syncCurrentRunnerFromModel(Net& emaModel) {
    std::scoped_lock lk(g_modelMutex, g_trtMutex);
    torch::NoGradGuard ng;
    return trtRefitFromTorchModel(g_trt, emaModel);
}

static bool snapshotCurrentIntoOld(Net& currentEmaModel,
    Net& oldModel,
    const std::string& planFile) {
        {
            std::lock_guard<std::mutex> lk(g_modelMutex);
            oldModel->to(torch::kCPU);
            copyNetState(oldModel, currentEmaModel);
            oldModel->eval();
        }

        if (!ensureOldRunnerReady(planFile)) return false;

        {
            std::lock_guard<std::mutex> lk(g_trtOldMutex);
            torch::NoGradGuard ng;
            return trtRefitFromTorchModel(g_trt_old, oldModel);
        }
}

static inline bool isFiniteF(float x) {
    return std::isfinite((double)x) != 0;
}











static void initAllOrExit(Net& model,
    Net& emaModel,
    const std::string& ptFile,
    const std::string& emaFile,
    const std::string& planFile) {
    setlocale(LC_ALL, "Russian");

    initDiceTable();
    initEpMaskAndNewDice();
    initZobrist();
    initLeaperAttacks();
    initNNConstPlanes();

#if defined(_MSC_VER) || defined(__x86_64__) || defined(__i386)
    bool wantPext = shouldUsePextPolicy();
    g_usePext = wantPext && (HAVE_PEXT_INTRIN != 0);
#else
    g_usePext = false;
#endif

    if (g_usePext) initSlidersPext();
    else          initSlidersMagics();

    if (!loadOrCreateTorchModel(ptFile, model)) {
        std::cerr << "Failed to load/create " << ptFile << "\n";
        std::exit(1);
    }

    if (!loadOrCreateEmaModel(emaFile, emaModel, model)) {
        std::cerr << "Failed to load/create " << emaFile << "\n";
        std::exit(1);
    }

    {
        std::lock_guard<std::mutex> lk(g_trtMutex);
        if (!g_trt.initOrCreate(planFile)) {
            std::cerr << "TensorRT: failed to initialize engine.\n";
            std::exit(1);
        }
        g_trtReady = true;
        g_nnBatch = TRT_MAX_BATCH;
    }

    // Initial refit
    {
        std::scoped_lock lk(g_modelMutex, g_trtMutex);
        torch::NoGradGuard ng;

        if (!trtRefitFromTorchModel(g_trt, emaModel)) {
            std::cerr << "TRT refit from net_ema.pt failed at startup.\n";
        }
    }
    //std::cerr << "[TRT] AI_HAVE_CUDA_KERNELS=" << AI_HAVE_CUDA_KERNELS << "\n";
}

static void saveAll(const std::string& ptFile,
    const std::string& emaFile,
    const std::string& planFile,
    const std::string& optFile,
    Net& model,
    Net& emaModel,
    Trainer& trainer) {
        {
            std::lock_guard<std::mutex> lk(g_modelMutex);

            try {
                torch::save(model, ptFile);
            }
            catch (const std::exception& e) {
                std::cerr << "torch::save(model) failed: " << e.what() << "\n";
            }

            try {
                torch::save(emaModel, emaFile);
            }
            catch (const std::exception& e) {
                std::cerr << "torch::save(emaModel) failed: " << e.what() << "\n";
            }

            if (!saveOptimizerState(optFile, trainer)) {
                std::cerr << "save optimizer state failed.\n";
            }
        }

        {
            std::lock_guard<std::mutex> lk(g_trtMutex);
            if (!trtSavePlanToDisk(g_trt, planFile)) {
                std::cerr << "TRT plan serialize failed.\n";
            }
        }
}

// ------------------------------------------------------------
// Training(minutes)
// ------------------------------------------------------------

static AI_FORCEINLINE bool waitForNoInferenceInFlightBounded(int64_t timeoutMs) {
    using Clock = std::chrono::steady_clock;
    const auto deadline = Clock::now() + std::chrono::milliseconds(timeoutMs);
    while (g_inferInFlight.load(std::memory_order_acquire) != 0) {
        if (Clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    return true;
}

// Bounded quiesce barrier. Returns false (and logs) instead of hanging forever
// when the shared server never drains; the caller skips its action this round.
static bool safeRefitBarrierShared(SharedInferenceServerTrain& srv) {
    if (!srv.waitIdleFor(120000)) {
        std::cerr << "[barrier] TIMEOUT: train server not idle after 120s"
            << " (queue=" << srv.size() << "); skipping quiesce round." << std::endl;
        return false;
    }
    if (!waitForNoInferenceInFlightBounded(120000)) {
        std::cerr << "[barrier] TIMEOUT: inference in flight after 120s (inFlight="
            << g_inferInFlight.load(std::memory_order_relaxed)
            << "); skipping quiesce round." << std::endl;
        return false;
    }
    srv.clearQueueUnsafeWhenIdle();
    if (!waitForNoInferenceInFlightBounded(120000)) {
        std::cerr << "[barrier] TIMEOUT after clearQueue; skipping quiesce round." << std::endl;
        return false;
    }
    return true;
}

static AI_FORCEINLINE bool tryClaimGameBudget(std::atomic<int>& gamesLeft) {
    int cur = gamesLeft.load(std::memory_order_relaxed);
    while (cur > 0) {
        if (gamesLeft.compare_exchange_weak(
            cur, cur - 1,
            std::memory_order_relaxed,
            std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

static AI_FORCEINLINE void refundGameBudget(std::atomic<int>& gamesLeft) {
    gamesLeft.fetch_add(1, std::memory_order_relaxed);
}

template<class Clock = std::chrono::steady_clock>
static AI_FORCEINLINE bool enoughTimeToStartNewGame(
    typename Clock::time_point deadline,
    std::chrono::milliseconds guard)
{
    return Clock::now() + guard < deadline;
}

struct SelfPlayBlockStats {
    uint64_t games = 0;
    uint64_t plies = 0;
    uint64_t truncated = 0;
    uint64_t samples = 0;
};

struct GameDurationStatsSnapshot {
    uint64_t games = 0;
    double meanMs = 0.0;
    double stddevMs = 0.0;
    double minMs = 0.0;
    double maxMs = 0.0;
};

struct GameDurationStats {
    mutable std::mutex m;
    uint64_t games = 0;

    // Welford online stats
    double meanMs = 0.0;
    double m2Ms = 0.0;

    double minMs = 0.0;
    double maxMs = 0.0;

    void reset() {
        std::lock_guard<std::mutex> lk(m);
        games = 0;
        meanMs = 0.0;
        m2Ms = 0.0;
        minMs = 0.0;
        maxMs = 0.0;
    }

    template<class Duration>
    void add(Duration d) {
        const double ms = std::chrono::duration<double, std::milli>(d).count();
        if (!(ms > 0.0) || !std::isfinite(ms)) return;

        std::lock_guard<std::mutex> lk(m);

        ++games;
        if (games == 1) {
            meanMs = ms;
            m2Ms = 0.0;
            minMs = ms;
            maxMs = ms;
            return;
        }

        const double delta = ms - meanMs;
        meanMs += delta / (double)games;
        const double delta2 = ms - meanMs;
        m2Ms += delta * delta2;

        if (ms < minMs) minMs = ms;
        if (ms > maxMs) maxMs = ms;
    }

    GameDurationStatsSnapshot snapshot() const {
        std::lock_guard<std::mutex> lk(m);

        GameDurationStatsSnapshot s;
        s.games = games;
        s.meanMs = meanMs;
        s.minMs = minMs;
        s.maxMs = maxMs;

        if (games >= 2) {
            const double var = std::max(0.0, m2Ms / (double)(games - 1));
            s.stddevMs = std::sqrt(var);
        }
        else {
            s.stddevMs = 0.0;
        }

        return s;
    }
};

static GameDurationStats g_selfPlayGameDurationStats;

static AI_FORCEINLINE std::chrono::milliseconds currentSelfPlayStartGuard() {
    using namespace std::chrono;

    constexpr auto kFallback = milliseconds(3000);
    constexpr auto kMin = milliseconds(500);
    constexpr auto kMax = milliseconds(30000);
    constexpr uint64_t kMinSamples = 8;
    constexpr double kSigmaMul = 0.5;

    const auto s = g_selfPlayGameDurationStats.snapshot();
    double guardMs = (double)kFallback.count();

    if (s.games >= kMinSamples && std::isfinite(s.meanMs) && std::isfinite(s.stddevMs)) {
        guardMs = s.meanMs + kSigmaMul * s.stddevMs;
        if (guardMs < s.meanMs) guardMs = s.meanMs;
    }

    if (!(guardMs > 0.0) || !std::isfinite(guardMs)) {
        guardMs = (double)kFallback.count();
    }

    auto guard = milliseconds((long long)std::llround(guardMs));

    if (guard < kMin) guard = kMin;
    if (guard > kMax) guard = kMax;
    return guard;
}

static AI_FORCEINLINE std::chrono::milliseconds
currentSelfPlayBlockDuration(std::chrono::milliseconds startGuard) {
    using namespace std::chrono;

    constexpr int kBlockToGuard = 20;

    long long ms = startGuard.count() * (long long)kBlockToGuard;
    if (ms <= 0) ms = 1;

    return milliseconds(ms);
}

static SearchPoolStatsSnapshot snapshotAllSearchStats(
    const std::vector<std::unique_ptr<GameContext>>& gamesCtx) {
    SearchPoolStatsSnapshot out{};
    for (const auto& sp : gamesCtx) {
        if (!sp) continue;
        auto s = sp->pool.snapshotStats();
        out.simsOk += s.simsOk;
        out.simsFail += s.simsFail;
        out.ttHit += s.ttHit;
        out.ttMiss += s.ttMiss;
        out.depthSum += s.depthSum;
    }
    return out;
}

static void runParallelSelfPlayBlock(
    std::vector<std::unique_ptr<GameContext>>& gamesCtx,
    ITrainInferenceServer& sharedSrv,
    BackendBinding backend,
    ReplayBuffer& rb,
    int simsPerPos,
    int maxPlies,
    bool addRootNoise,
    int maxGamesThisBlock,
    int gamesRemainingTotal,
    std::chrono::steady_clock::time_point deadline,
    std::chrono::milliseconds startGuard,
    SelfPlayBlockStats& outStats) {

    outStats = {};

    const int budget = std::max(0, std::min(maxGamesThisBlock, gamesRemainingTotal));
    if (budget <= 0 || gamesCtx.empty()) return;

    std::atomic<int> gamesLeft{ budget };
    std::atomic<uint64_t> gamesDone{ 0 };
    std::atomic<uint64_t> pliesDone{ 0 };
    std::atomic<uint64_t> truncatedDone{ 0 };
    std::atomic<uint64_t> samplesDone{ 0 };
    std::atomic<bool> abortAll{ false };

    std::mutex exM;
    std::exception_ptr ex;

    std::vector<std::thread> outer;
    outer.reserve(gamesCtx.size());

    for (size_t i = 0; i < gamesCtx.size(); ++i) {
        outer.emplace_back([&, i] {
            try {
                GameContext& sp = *gamesCtx[i];

                for (;;) {
                    using Clock = std::chrono::steady_clock;

                    if (abortAll.load(std::memory_order_relaxed)) break;

                    if (!enoughTimeToStartNewGame<Clock>(deadline, startGuard)) break;

                    if (!tryClaimGameBudget(gamesLeft)) break;

                    if (!enoughTimeToStartNewGame<Clock>(deadline, startGuard)) {
                        refundGameBudget(gamesLeft);
                        break;
                    }

                    int plyCount = 0;
                    bool terminated = false;
                    int samplesAdded = 0;

                    const auto gameT0 = Clock::now();

                    selfPlayOneGame960(
                        sp,
                        sharedSrv,
                        backend,
                        rb,
                        simsPerPos,
                        maxPlies,
                        addRootNoise,
                        plyCount,
                        terminated,
                        samplesAdded
                    );

                    const auto gameT1 = Clock::now();

                    if (!sp.T.abort.load(std::memory_order_relaxed) &&
                        (terminated || plyCount > 0 || samplesAdded > 0)) {
                        g_selfPlayGameDurationStats.add(gameT1 - gameT0);
                    }

                    gamesDone.fetch_add(1, std::memory_order_relaxed);
                    noteTrainingProgress();
                    pliesDone.fetch_add((uint64_t)std::max(0, plyCount), std::memory_order_relaxed);
                    samplesDone.fetch_add((uint64_t)std::max(0, samplesAdded), std::memory_order_relaxed);

                    if (!terminated && plyCount >= maxPlies) {
                        truncatedDone.fetch_add(1, std::memory_order_relaxed);
                    }

                    if (sp.T.abort.load(std::memory_order_relaxed)) {
                        std::cerr << "[selfplay] game_ctx=" << i
                            << " aborted: oomCode="
                            << sp.T.oomCode.load(std::memory_order_relaxed)
                            << " (not enough memory for "
                            << oomWhat(sp.T.oomCode.load(std::memory_order_relaxed))
                            << ") -> reset table\n";
                        sp.resetForNewGame();
                    }
                }
            }
            catch (...) {
                abortAll.store(true, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lk(exM);
                if (!ex) ex = std::current_exception();
            }
            });
    }

    for (auto& th : outer) {
        if (th.joinable()) th.join();
    }

    if (ex) std::rethrow_exception(ex);

    outStats.games = gamesDone.load(std::memory_order_relaxed);
    outStats.plies = pliesDone.load(std::memory_order_relaxed);
    outStats.truncated = truncatedDone.load(std::memory_order_relaxed);
    outStats.samples = samplesDone.load(std::memory_order_relaxed);
}

static std::string fmtCompactU64(uint64_t x) {
    std::ostringstream oss;
    if (x >= 1000000000ull) {
        oss << std::fixed << std::setprecision(1) << (double)x / 1e9 << "b";
    }
    else if (x >= 1000000ull) {
        oss << std::fixed << std::setprecision(1) << (double)x / 1e6 << "m";
    }
    else if (x >= 1000ull) {
        oss << std::fixed << std::setprecision(0) << (double)x / 1e3 << "k";
    }
    else {
        oss << x;
    }
    return oss.str();
}

static std::string fmtFixed(double x, int prec) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(prec) << x;
    return oss.str();
}

void Training(int targetGames) {
    diagLogLine("[Training] started, targetGames=" + std::to_string(targetGames));
    g_selfPlayGameDurationStats.reset();

    const std::string ptFile = "net.pt";
    const std::string emaFile = "net_ema.pt";
    const std::string planFile = "net.plan";
    const std::string optFile = "optimizer.pt";

    Net model;
    Net emaModel;
    initAllOrExit(model, emaModel, ptFile, emaFile, planFile);

    // Replay
    static constexpr size_t REPLAY_CAP = 1000000;
    ReplayBuffer rb(REPLAY_CAP);

    // Trainer: initialize from step zero, then try to restore optimizer
    Trainer trainer;

    trainer.init(model, emaModel);

    if (loadOptimizerState(optFile, trainer)) {
        std::cerr << "optimizer state restored.\n";

        // After optimizer restore, force-set LR again according to scheduler
        trainer.current_lr = -1.0;
        trainer.updateLR(true);
    }
    else {
        std::cerr << "no optimizer state found, starting fresh.\n";
    }

    Net oldModel;
    oldModel->to(torch::kCPU);
    oldModel->eval();

    if (!snapshotCurrentIntoOld(emaModel, oldModel, planFile)) {
        std::cerr << "[arena] failed to initialize old snapshot in memory.\n";
    }

    BackendBinding sharedBackend{ g_trt, g_trtMutex };
    SharedInferenceServerTrain sharedSrv(sharedBackend);
    sharedSrv.start();

    const unsigned hwSP = std::max(1u, std::thread::hardware_concurrency());

    // For training, keep exactly 1 search thread per game.
    // Then best throughput usually comes from more parallel games,
    // but without insane expansion of GameContext count.
    const unsigned SEARCH_THREADS_PER_GAME = 1u;

    const unsigned PARALLEL_GAMES = std::max(2u, hwSP - 4u);

    const size_t SP_NODE_POW2 =
        (PARALLEL_GAMES >= 6 ? (1u << 18) :
            PARALLEL_GAMES >= 4 ? (1u << 19) : (1u << 20));

    const size_t SP_EDGE_CAP =
        (PARALLEL_GAMES >= 6 ? (1u << 22) :
            PARALLEL_GAMES >= 4 ? (1u << 23) : (1u << 24));

    std::vector<std::unique_ptr<GameContext>> gamesCtx;
    gamesCtx.reserve(PARALLEL_GAMES);

    struct TrainingCleanupGuard {
        std::vector<std::unique_ptr<GameContext>>* games = nullptr;
        SharedInferenceServerTrain* sharedSrv = nullptr;

        ~TrainingCleanupGuard() noexcept {
            try {
                if (games) {
                    for (auto& g : *games) {
                        if (g) g->stop();
                    }
                }
            }
            catch (...) {}

            try {
                if (sharedSrv) {
                    sharedSrv->requestStop();
                    sharedSrv->join();
                }
            }
            catch (...) {}
        }
    } cleanupGuard{ &gamesCtx, &sharedSrv };

    for (unsigned i = 0; i < PARALLEL_GAMES; ++i) {
        auto g = std::make_unique<GameContext>(SP_NODE_POW2, SP_EDGE_CAP);
        g->start(SEARCH_THREADS_PER_GAME);
        gamesCtx.push_back(std::move(g));
    }

    bool spRunning = true;
    SearchPoolStatsSnapshot prevSearchStats = snapshotAllSearchStats(gamesCtx);
    bool stopTraining = false;

    std::cout << "[selfplay] parallel_games=" << PARALLEL_GAMES << "\n";

    // -------------------------------
    // SCHEDULER
    // -------------------------------
    static constexpr double REPLAY_RATIO = 6.0;          // consumed / added
    static constexpr int TRAIN_MAX_STEPS_PER_BLOCK = 9999;
    static constexpr int TRAIN_WARMUP_BATCHES = 1000;

    const int simsPerPos = 800;
    const int maxPlies = 256;
    const bool addRootNoise = true;

    const size_t MIN_REPLAY_TO_TRAIN =
        (size_t)trainer.B * (size_t)TRAIN_WARMUP_BATCHES;

    bool trainSchedulerActive = false;
    double trainSampleCredits = 0.0; // measured in "samples to consume"

    auto t0 = std::chrono::steady_clock::now();
    auto nextSave = t0 + std::chrono::hours(1);
    auto nextStat = t0 + std::chrono::seconds(10);

    int games = 0;
    int refits = 0;
    uint64_t statGamesWindow = 0;
    uint64_t statPlyWindow = 0;
    uint64_t statTruncatedWindow = 0;
    auto statWindowStart = std::chrono::steady_clock::now();
    int nextArenaAt = 100000;

    // ---- fast-restart state: replay buffer + counters ----
    {
        const size_t restored = rb.loadFromFile("replay.bin");
        if (restored) {
            std::cout << "[replay] restored " << restored
                << " samples from replay.bin" << std::endl;
        }
        std::ifstream st("train_state.txt");
        if (st) {
            long long g = 0; double credits = 0.0;
            if (st >> g >> credits) {
                games = (int)std::max(0ll, g);
                trainSampleCredits = std::max(0.0, credits);
                while (games >= nextArenaAt) nextArenaAt += 100000;
                std::cout << "[state] restored games=" << games
                    << " credits=" << trainSampleCredits << std::endl;
            }
        }
    }

    // Arena gate #2: require at least 100k games played by THIS process,
    // so a fresh restart never triggers an arena almost immediately.
    const int gamesAtProcessStart = games;

    // Arenas run only at 100k-multiples of the TOTAL game counter; skip any
    // milestone that would fall earlier than start+100k (missed ones are dropped).
    while (nextArenaAt < gamesAtProcessStart + 100000) nextArenaAt += 100000;

    // Network snapshots, unlike arenas, happen at EVERY 100k-multiple reached.
    int nextNetSaveAt = (games / 100000 + 1) * 100000;

    // Deadlock watchdog: if nothing progresses for STALL_LIMIT, dump diagnostics
    // and exit(42). State is persisted hourly, so the supervisor restarts cheaply.
    noteTrainingProgress();
    std::atomic<bool> watchdogStop{ false };
    std::thread watchdogTh([&] {
        static constexpr uint64_t STALL_LIMIT_MS = 15ull * 60ull * 1000ull;
        while (!watchdogStop.load(std::memory_order_relaxed)) {
            for (int i = 0; i < 30 && !watchdogStop.load(std::memory_order_relaxed); ++i) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
            if (watchdogStop.load(std::memory_order_relaxed)) break;

            const uint64_t last = g_progressHeartbeatMs.load(std::memory_order_relaxed);
            const uint64_t now = steadyNowMs();
            if (last != 0 && now > last && (now - last) > STALL_LIMIT_MS) {
                std::ostringstream oss;
                oss << "[watchdog] no progress for " << ((now - last) / 1000)
                    << "s; nnInFlight=" << g_inferInFlight.load(std::memory_order_relaxed)
                    << " failGetNode=" << g_failGetNode.load(std::memory_order_relaxed)
                    << " failExpandWait=" << g_failExpandWait.load(std::memory_order_relaxed)
                    << " failDepth=" << g_failDepth.load(std::memory_order_relaxed)
                    << " -> exit(42) for supervised restart";
                std::cerr << oss.str() << std::endl;
                diagLogLine(oss.str());
                std::cerr.flush();
                std::cout.flush();
                std::_Exit(42);
            }
        }
        });
    struct WatchdogJoin {
        std::atomic<bool>* flag; std::thread* th;
        ~WatchdogJoin() noexcept {
            flag->store(true, std::memory_order_relaxed);
            if (th->joinable()) th->join();
        }
    } watchdogJoin{ &watchdogStop, &watchdogTh };

    std::cout << "Starting training for " << targetGames << " games...\n";

    while (games < targetGames) {
        // ===========================
        // 1) SELF-PLAY BLOCK
        // ===========================
        const auto startGuard = currentSelfPlayStartGuard();
        const auto selfPlayBlockDur = currentSelfPlayBlockDuration(startGuard);
        const auto spEnd = std::chrono::steady_clock::now() + selfPlayBlockDur;

        SelfPlayBlockStats spBlk;
        runParallelSelfPlayBlock(
            gamesCtx,
            sharedSrv,
            sharedBackend,
            rb,
            simsPerPos,
            maxPlies,
            addRootNoise,
            targetGames - games,
            targetGames - games,
            spEnd,
            startGuard,
            spBlk
        );

        games += (int)spBlk.games;
        noteTrainingProgress();

        // Versioned network snapshot at every 100k-multiple of the total game
        // counter, independent of whether an arena runs at that milestone.
        while (games >= nextNetSaveAt) {
            const int netVer = nextNetSaveAt / 100000;
            const std::string vp = "net" + std::to_string(netVer);
            try {
                std::lock_guard<std::mutex> lkM(g_modelMutex);
                torch::save(model, vp + ".pt");
                torch::save(emaModel, vp + "_ema.pt");
            }
            catch (const std::exception& e) {
                std::cerr << "[snapshot] save failed for " << vp << ": " << e.what() << std::endl;
            }
            {
                std::lock_guard<std::mutex> lkT(g_trtMutex);
                if (!trtSavePlanToDisk(g_trt, vp + ".plan")) {
                    std::cerr << "[snapshot] plan save failed for " << vp << std::endl;
                }
            }
            std::cout << "[snapshot] saved " << vp << ".pt, " << vp << "_ema.pt, "
                << vp << ".plan at " << nextNetSaveAt << " games." << std::endl;
            nextNetSaveAt += 100000;
        }

        statGamesWindow += spBlk.games;
        statPlyWindow += spBlk.plies;
        statTruncatedWindow += spBlk.truncated;

        if (!trainSchedulerActive && rb.currentSize() >= MIN_REPLAY_TO_TRAIN) {
            trainSchedulerActive = true;
            std::cerr << "[trainer] replay warmup reached: "
                << rb.currentSize()
                << " samples, sample-based schedule enabled\n";
        }

        if (trainSchedulerActive && spBlk.samples > 0) {
            trainSampleCredits += REPLAY_RATIO * (double)spBlk.samples;
        }

        // ===========================
        // 2) TRAIN BLOCK (sample-based, replay ratio = 6)
        // ===========================
        int didTrain = 0;

        if (trainSchedulerActive) {
            const int targetSteps =
                std::min(TRAIN_MAX_STEPS_PER_BLOCK,
                    (int)(trainSampleCredits / (double)trainer.B));

            if (targetSteps > 0) {
                if (safeRefitBarrierShared(sharedSrv)) {
                    // use old function as fixed-step runner by giving it a huge time budget
                    didTrain = trainer.trainBlockBudgetMs(rb, model, emaModel,
                        /*budgetMs=*/24 * 60 * 60 * 1000,
                        /*maxStepsHard=*/targetSteps,
                        TRAIN_WARMUP_BATCHES);

                    trainSampleCredits -= (double)didTrain * (double)trainer.B;
                    if (trainSampleCredits < 0.0) trainSampleCredits = 0.0;
                }
                else {
                    std::cerr << "[trainer] train block skipped (barrier timeout)." << std::endl;
                }
            }
        }

        // ===========================
        // 3) REFIT TRT
        // ===========================
        if (didTrain > 0) {
            if (safeRefitBarrierShared(sharedSrv)) {
                std::scoped_lock lk(g_modelMutex, g_trtMutex);
                torch::NoGradGuard ng;

                if (!trtRefitFromTorchModel(g_trt, emaModel)) {
                    std::cerr << "[refit] TRT refit failed.\n";
                }
                else {
                    ++refits;
                }
            }
            else {
                std::cerr << "[refit] skipped (barrier timeout)." << std::endl;
            }
        }

        while (games >= nextArenaAt &&
            (games - gamesAtProcessStart) >= 100000) {
            // Fully pause main self-play during arena:
            // remove extra worker threads / inference server activity.
            if (spRunning) {
                safeRefitBarrierShared(sharedSrv);
                for (auto& g : gamesCtx) {
                    if (g) g->stop();
                }
                spRunning = false;
            }

            // Full hourly-style save before the (multi-hour) arena, so a crash
            // or stop during the arena loses nothing.
            saveAll(ptFile, emaFile, planFile, optFile, model, emaModel, trainer);
            if (rb.saveToFile("replay.bin")) {
                std::ofstream st("train_state.txt", std::ios::trunc);
                st << games << ' ' << trainSampleCredits << '\n';
                std::cout << "[arena] pre-arena state saved (replay.bin, "
                    << rb.currentSize() << " samples)." << std::endl;
            }
            else {
                std::cerr << "[arena] pre-arena replay.bin save FAILED." << std::endl;
            }
            nextSave = std::chrono::steady_clock::now() + std::chrono::hours(1);

            bool arenaOk = true;

            // current TRT must exactly match the current EMA model
            if (!syncCurrentRunnerFromModel(emaModel)) {
                std::cerr << "[arena] failed to sync current TRT from EMA model.\n";
                arenaOk = false;
            }

            if (arenaOk && !g_trtOldReady) {
                if (!snapshotCurrentIntoOld(emaModel, oldModel, planFile)) {
                    std::cerr << "[arena] failed to prepare old TRT snapshot.\n";
                    arenaOk = false;
                }
            }

            if (arenaOk) {
                // (Network snapshot for this milestone is already written by the
                // unconditional 100k-multiple snapshot in the self-play section.)
                static constexpr int ARENA_GAMES = 10000;
                std::cout << "\n[arena] start: current vs old, games=" << ARENA_GAMES
                    << ", sims=800, triggerGames=" << nextArenaAt << std::endl;

                ArenaStats ar = runArenaMatch(/*games=*/ARENA_GAMES, /*simsPerPos=*/800);

                const double arenaLos = computeLOSPercent(ar.curWins, ar.oldWins);
                std::cout << "[arena] games=" << ARENA_GAMES << " W/L="
                    << ar.curWins << "/" << ar.oldWins
                    << " score=" << std::fixed << std::setprecision(4) << ar.currentScore()
                    << " LOS=" << std::setprecision(2) << arenaLos << "%\n";

                // promotion rule: if current > old, old := current
                if (ar.currentScore() > 0.5) {
                    if (snapshotCurrentIntoOld(emaModel, oldModel, planFile)) {
                        std::cout << "[arena] promoted current EMA -> old snapshot in memory\n";
                    }
                    else {
                        std::cerr << "[arena] promotion failed\n";
                    }
                }
                else {
                    std::cout << "[arena] old snapshot kept\n";
                }
            }
            else {
                std::cerr << "[arena] skipped due to setup failure.\n";
            }

            // IMPORTANT: always resume main self-play after arena.
            for (auto& g : gamesCtx) {
                if (g) g->start(SEARCH_THREADS_PER_GAME);
            }
            spRunning = true;
            prevSearchStats = snapshotAllSearchStats(gamesCtx);
            statWindowStart = std::chrono::steady_clock::now();

            // Even if arena setup failed, do not get stuck on the same threshold.
            nextArenaAt += 100000;

            if (!arenaOk) {
                break;
            }
        }

        if (stopTraining) break;

        // ===========================
        // 4) SAVE / STATS
        // ===========================
        auto now = std::chrono::steady_clock::now();

        if (now >= nextSave) {
            safeRefitBarrierShared(sharedSrv);
            nextSave += std::chrono::hours(1);

            saveAll(ptFile, emaFile, planFile, optFile, model, emaModel, trainer);

            // Fast-restart state: replay buffer + counters (atomic replace via .tmp).
            if (rb.saveToFile("replay.bin")) {
                std::ofstream st("train_state.txt", std::ios::trunc);
                st << games << ' ' << trainSampleCredits << '\n';
                std::cout << "[autosave] replay.bin saved ("
                    << rb.currentSize() << " samples)." << std::endl;
            }
            else {
                std::cerr << "[autosave] replay.bin save FAILED." << std::endl;
            }

            std::cout << "[autosave] Progress: " << games << " / " << targetGames << " games." << std::endl;
        }

        if (now >= nextStat) {
            nextStat += std::chrono::seconds(10);

            auto curStats = snapshotAllSearchStats(gamesCtx);
            auto dtSec = std::chrono::duration<double>(now - statWindowStart).count();
            if (dtSec <= 0.0) dtSec = 1e-9;

            const uint64_t dSimsOk = curStats.simsOk - prevSearchStats.simsOk;
            const uint64_t dSimsFail = curStats.simsFail - prevSearchStats.simsFail;
            const uint64_t dTTHit = curStats.ttHit - prevSearchStats.ttHit;
            const uint64_t dTTMiss = curStats.ttMiss - prevSearchStats.ttMiss;
            const uint64_t dDepth = curStats.depthSum - prevSearchStats.depthSum;

            const double nps = (double)dSimsOk / dtSec;
            const double ttHitPct = (dTTHit + dTTMiss)
                ? (100.0 * (double)dTTHit / (double)(dTTHit + dTTMiss))
                : 0.0;
            const double avgDepth = dSimsOk
                ? ((double)dDepth / (double)dSimsOk)
                : 0.0;

            const double avgLen = statGamesWindow
                ? ((double)statPlyWindow / (double)statGamesWindow)
                : 0.0;
            const double truncatedPct = statGamesWindow
                ? (100.0 * (double)statTruncatedWindow / (double)statGamesWindow)
                : 0.0;

            const double elapsedSecTotal =
                std::chrono::duration<double>(now - t0).count();
            const double nnCallsPerSec = (elapsedSecTotal > 1e-9)
                ? ((double)g_inferBatchCount.load(std::memory_order_relaxed) / elapsedSecTotal)
                : 0.0;
            const double nnDutyPct = (elapsedSecTotal > 1e-9)
                ? std::clamp(
                    (100.0 * (double)g_inferBusyMicros.load(std::memory_order_relaxed))
                    / (elapsedSecTotal * 1.0e6),
                    0.0, 100.0)
                : 0.0;

            double remainDays = 0.0;
            bool haveEta = false;

            if (games > 0 && elapsedSecTotal > 1.0) {
                const double gamesPerSecTotal = (double)games / elapsedSecTotal;
                if (gamesPerSecTotal > 1e-9) {
                    const double gamesLeft = (double)std::max(0, targetGames - games);
                    remainDays = (gamesLeft / gamesPerSecTotal) / 86400.0;
                    if (std::isfinite(remainDays)) {
                        haveEta = true;
                    }
                }
            }
            float b = stof(fmtFixed(getAverageInferBatchSize(), 2));
            std::cout << "Time: ";
            if (haveEta) std::cout << fmtFixed(remainDays, 2);
            else         std::cout << "--";

            std::cout
                << " | Games: " << games
                << " | Replay: " << fmtCompactU64((uint64_t)rb.currentSize())
                << " | Step: " << trainer.steps
                << " | P: " << fmtFixed(trainer.lastLossP, 2)
                << " | V: " << fmtFixed(trainer.lastVMAE, 2)
                << " | Grad: " << fmtFixed(trainer.lastGradNorm, 1)
                << " | Len: " << fmtFixed(avgLen, 1)
                << " | NPS: " << fmtFixed(nps, 0)
                << " | Batch: " << b
                << " | Duty: " << fmtFixed(nnDutyPct, 1) << "%"
                << " | Speed: " << stof(fmtFixed(nnCallsPerSec, 1)) * b
                << " | Depth: " << fmtFixed(avgDepth, 0)
                << "\n";

            prevSearchStats = curStats;
            statWindowStart = now;
            statGamesWindow = 0;
            statPlyWindow = 0;
            statTruncatedWindow = 0;
            (void)dSimsFail;
        }
    }

    // ==========================================
    // 5) CLEAN STOP & FINAL SAVE + FINAL REBUILD
    // ==========================================
    if (spRunning) {
        safeRefitBarrierShared(sharedSrv);
        for (auto& g : gamesCtx) {
            if (g) g->stop();
        }
        spRunning = false;
    }

    sharedSrv.requestStop();
    sharedSrv.join();

    std::cout << "\n[Completion] Collected " << targetGames << " games. Saving final weights...\n";
    if (rb.saveToFile("replay.bin")) {
        std::ofstream st("train_state.txt", std::ios::trunc);
        st << games << ' ' << trainSampleCredits << '\n';
    }
    {
        std::lock_guard<std::mutex> lk(g_modelMutex);

        try {
            torch::save(model, ptFile);
        }
        catch (const std::exception& e) {
            std::cerr << "torch::save(model) failed: " << e.what() << "\n";
        }

        try {
            torch::save(emaModel, emaFile);
        }
        catch (const std::exception& e) {
            std::cerr << "torch::save(emaModel) failed: " << e.what() << "\n";
        }

        if (!saveOptimizerState(optFile, trainer)) {
            std::cerr << "final save optimizer state failed.\n";
        }

    }

    std::cout << "[Completion] Starting final TensorRT rebuild. This will take a couple of minutes...\n";

    // 1) shutdown + remove old plan
    {
        std::lock_guard<std::mutex> lkT(g_trtMutex);
        g_trt.shutdown();
        g_trtReady = false;
        if (::remove(planFile.c_str()) != 0) {
            // not fatal: file may not have existed
        }
    }

    // 2) rebuild/load new plan
    bool okInit = false;
    {
        std::lock_guard<std::mutex> lkT(g_trtMutex);
        okInit = g_trt.initOrCreate(planFile);
    }

    if (!okInit) {
        std::cerr << "[Completion] FATAL ERROR: Failed to rebuild the final net.plan!\n";
    }
    else {
        // 3) refit from final torch model and save final plan
        {
            std::scoped_lock lk(g_modelMutex, g_trtMutex);
            torch::NoGradGuard ng;

            if (trtRefitFromTorchModel(g_trt, emaModel)) {
                trtSavePlanToDisk(g_trt, planFile);
            }
        }

        // 4) final TRT shutdown
        {
            std::lock_guard<std::mutex> lkT(g_trtMutex);
            g_trt.shutdown();
            g_trtReady = false;
        }
    }

    {
        std::lock_guard<std::mutex> lk(g_trtOldMutex);
        g_trt_old.shutdown();
        g_trtOldReady = false;
    }

    std::cout << "Training completed successfully! Files net.pt, net_ema.pt, optimizer.pt, and net.plan are ready.\n";
    diagLogLine("[Training] finished normally");
}
vector<long long> sqKey = { 100002324,505950000,609121555,110242252,614571640,610172920,35020000,355080122,34360208,37770000,338410393,631600651,0 };
vector<long long> diceKey = { 100001856,405160010,6571113,6831247,309371046,606462162,324110006,341110119,126520169,123350009,430810277,826600559,188800000000,285506010000,115807560000,127107610000,110310990000,220107560000,1124640000,14842670000,26427290000,5623910000,33132030000,64527490000 };
vector<int> stabKey = { -9495531,-9561067,-10020846,-10086382,-13169399,-13169655,-13432313 };
struct BOARDSTAT { vector<int> board; int num; int time; };
int NUMBER(long long key, vector<long long>& v) {
    int min, i, d, n;
    min = INT_MAX;
    for (i = 0; i < v.size(); i++) {
        d = abs(v[i] % 10000 - key % 10000) + abs(v[i] / 10000 % 10000 - key / 10000 % 10000) + abs(v[i] / 100000000 - key / 100000000);
        if (d < min) {
            n = i;
            min = d;
        }
    }
    return n;
}
vector<int> S(int x1, int x2, int y1, int y2) {
    int w, h;
    vector<int> s;
    HDC d, m;
    HBITMAP b;
    BITMAPINFO p;
    w = x2 - x1 + 1;
    h = y2 - y1 + 1;
    s.resize(w * h);
    d = GetDC(0);
    m = CreateCompatibleDC(d);
    b = CreateCompatibleBitmap(d, w, h);
    p = { 40,w,-h,1,32 };
    SelectObject(m, b);
    BitBlt(m, 0, 0, w, h, d, x1, y1, 13369376);
    GetDIBits(d, b, 0, h, s.data(), &p, 0);
    DeleteObject(b);
    DeleteObject(m);
    DeleteObject(d);
    return s;
}
vector<int> S() {
    vector<int> s;
    s = S(407, 1510, 550, 1865);
    if (s[928 + 1104 * 430] == -15189205)s = {};
    return s;
}
int FLIP(vector<int>& s) { return s.size() && s[8 + 1104 * 1200] == -665935; }
int SIDE(vector<int>& s) { return s.size() && s[1038 + 1104 * 40] == -16443635 != FLIP(s); }
vector<int> BOARD(vector<int>& s) {
    int sq, SQ, x, y, p;
    long long key;
    vector<int> board;
    if (s.empty())return {};
    for (sq = 0; sq < 64; sq++) {
        if (FLIP(s) == 0)SQ = sq ^ 56; else SQ = sq ^ 7;
        key = 0;
        for (x = 0; x < 138; x++)for (y = 0; y < 138; y++) {
            p = s[138 * (SQ % 8) + x + 1104 * (212 + 138 * (SQ / 8) + y)];
            key += (p == -1) + 10000 * (p == -16777216) + 100000000 * (p == -8421505);
        }
        board.push_back(NUMBER(key, sqKey));
    }
    return board;
}
vector<int> DICE(vector<int>& s) {
    int i, x, y, p;
    long long key;
    vector<int> dice;
    if (s.empty())return {};
    for (i = 0; i < 3; i++) {
        key = 0;
        for (x = 0; x < 158; x++)for (y = 0; y < 158; y++) {
            p = s[248 + 228 * i + x + 1104 * y];
            key += (p == -1) + 10000 * (p == -16777216) + 100000000 * (p == -8421505);
        }
        dice.push_back(NUMBER(key, diceKey));
    }
    return dice;
}
int FROM(int sq) { return sq == 4 || sq >= 24 && sq <= 39 || sq == 60; }
vector<int> WAY(vector<int>& b1, vector<int>& b2) {
    int sq;
    vector<int> way;
    if (b1.empty() || b2.empty())return {};
    for (sq = 0; sq < 64; sq++)if (b2[sq] != b1[sq])way.push_back(sq);
    sort(way.begin(), way.end(), [&](int a, int b) {return b2[a] > b2[b] || b2[a] == b2[b] && FROM(a) > FROM(b); });
    return way;
}
void START(Position& pos) {
    pos.color = { 0,0 };
    pos.piece = { 0,0,0,0,0,0 };
    pos.side = 0;
    pos.ep1 = { 0,0 };
    pos.ep2 = 0;
    pos.rook = { 0,7,56,63 };
    pos.castle = 15;
    pos.dice = 0;
    pos.key = computeKey(pos);
}
void START() {
    PATH = { bit(1) | bit(2) | bit(3),bit(5) | bit(6),bit(57) | bit(58) | bit(59),bit(61) | bit(62) };
    MASK.fill(0);
    MASK[0] = 1;
    MASK[4] = 3;
    MASK[7] = 2;
    MASK[56] = 4;
    MASK[60] = 12;
    MASK[63] = 8;
}
void BOARDSET(vector<int>& board) {
    int sq, piece;
    POS.color = { 0,0 };
    POS.piece = { 0,0,0,0,0,0 };
    if (board.empty())return;
    for (sq = 0; sq < 64; sq++) {
        piece = board[sq];
        if (piece == 12)continue;
        POS.color[piece / 6] |= bit(sq);
        POS.piece[piece % 6] |= bit(sq);
    }
}
void CASTLESET() {
    int c, r;
    for (c = 0; c < 2; c++) {
        for (r = 0; r < 2; r++)if ((POS.color[c] & POS.piece[3] & bit(7 * r + 56 * c)) == 0)POS.castle &= ~bit(r + 2 * c);
        if ((POS.color[c] & POS.piece[5] & bit(4 + 56 * c)) == 0)POS.castle &= ~(bit(2 * c) | bit(1 + 2 * c));
    }
}
void DICESET(vector<int>& s) {
    int i, dice, d;
    uint64_t pawns;
    string t;
    vector<int> v;
    if (s.empty()) {
        POS.dice = 0;
        return;
    }
    v = DICE(s);
    for (i = 0; i < 3; i++)v[i] %= 6;
    sort(v.begin(), v.end());
    for (i = 0; i < 3; i++)t += pieceChar(v[i]);
    dice = diceFenToInt(t);
    pawns = POS.color[POS.side] & POS.piece[0];
    d = 6;
    if (pawns)if (POS.side == 0)d = clz64(pawns) >> 3; else d = ctz64(pawns) >> 3;
    for (i = 0; i < 5; i++)while (dicePiece[dice][i] && (POS.color[POS.side] & POS.piece[i]) == 0 && d > dicePiece[dice][0])dice = newDice[dice][i];
    POS.dice = dice;
}
void END(vector<int>& s1, vector<int>& s2, vector<int>& b1, vector<int>& b2) {
    s1 = s2;
    b1 = b2;
}
vector<int> STAB(vector<int>& s) {
    int i, n;
    vector<int> stab;
    if (s.empty())return { 2,2,2 };
    for (i = 0; i < 3; i++) {
        n = find(stabKey.begin(), stabKey.end(), s[326 + 228 * i + 1104 * 147]) - stabKey.begin();
        stab.push_back(n / 4 + (n == 7));
    }
    return stab;
}
int STABMIN(vector<int>& s1, vector<int>& s2) {
    int dark, act, i;
    act = dark = 0;
    for (i = 0; i < 3; i++)if (STAB(s2)[i] != 1)act = 1; else if (STAB(s1)[i] == 2)dark = 1;
    return dark && act;
}
int STABFULL(vector<int>& s) {
    int w, i;
    vector<int> stab;
    stab = STAB(s);
    w = 0;
    for (i = 0; i < 3; i++) {
        if (stab[i] == 2)return 0;
        if (stab[i] == 0)w = 1;
    }
    return w;
}
int STABFULL(vector<int>& s1, vector<int>& s2) { return STABFULL(s2) > STABFULL(s1); }
int PROMO(int piece, int sq) { return piece == 0 && sq / 8 == 7 || piece == 6 && sq / 8 == 0; }
int BOARDNEXT(vector<int>& b1, vector<int>& b2) {
    int side, x, dir, i;
    vector<int> way, key;
    vector<vector<int>> v;
    for (side = 0; side < 2; side++)for (x = 0; x < 8; x++)for (dir = -1; dir <= 1; dir += 2)if (x + dir >= 0 && x + dir <= 7)v.push_back({ x + 32 - 8 * side,6 * side,12,x + dir + 32 - 8 * side,6 * !side,12,x + dir + 40 - 24 * side,12,6 * side });
    for (side = 0; side < 2; side++)for (dir = 0; dir < 2; dir++)v.push_back({ 4 + 56 * side,5 + 6 * side,12,7 * dir + 56 * side,3 + 6 * side,12,2 + 4 * dir + 56 * side,12,5 + 6 * side,3 + 2 * dir + 56 * side,12,3 + 6 * side });
    way = WAY(b1, b2);
    for (i = 0; i < way.size(); i++) {
        key.push_back(way[i]);
        key.push_back(b1[way[i]]);
        key.push_back(b2[way[i]]);
    }
    return key.size() == 6 && key[2] == 12 && (PROMO(key[1], key[3]) == 0 && key[5] == key[1] || PROMO(key[1], key[3]) && key[5] >= key[1] + 1 && key[5] <= key[1] + 4) || find(v.begin(), v.end(), key) < v.end();
}
void ADD(vector<int>& b1, vector<int>& b, vector<BOARDSTAT>& bs) {
    int i;
    if (BOARDNEXT(b1, b) == 0)return;
    for (i = 0; i < bs.size(); i++)if (bs[i].board == b)break;
    if (i < bs.size()) {
        bs[i].num++;
        bs[i].time = clock();
    }
    else bs.push_back({ b,1,clock() });
    sort(bs.begin(), bs.end(), [](BOARDSTAT a, BOARDSTAT b) {return a.num > b.num || a.num == b.num && a.time > b.time; });
}
int DICENEXT(vector<int>& s1, vector<int>& s2) {
    int dark, i, dif;
    vector<int> d1, d2;
    if (s1.empty() || s2.empty())return 0;
    d1 = DICE(s1);
    d2 = DICE(s2);
    dark = 0;
    for (i = 0; i < 3; i++) {
        dif = d2[i] - d1[i];
        if (dif % 12)return 0;
        if (dif == 12)dark = 1;
    }
    return dark;
}
int DIF(int a, int b) {
    a += 16777216;
    b += 16777216;
    return abs(b % 256 - a % 256) + abs(b / 256 % 256 - a / 256 % 256) + abs(b / 65536 - a / 65536);
}
int DIF(vector<int>& s1, vector<int>& s2) {
    int dif, i, x, y, n;
    if (s1.empty() || s2.empty())return 1;
    dif = 0;
    for (i = 0; i < 3; i++)for (x = 0; x < 158; x++)for (y = 0; y < 158; y++) {
        n = 248 + 228 * i + x + 1104 * y;
        dif += DIF(s1[n], s2[n]);
    }
    return dif >= 10000;
}
void NEW(int& roll, int& change, vector<int>& s1, vector<int>& s2, vector<int>& b1, vector<int>& b2) {
    int stabmin, stabfull, i;
    time_point<steady_clock> t1, t2;
    vector<int> b;
    vector<vector<int>> v;
    vector<BOARDSTAT> bs;
    t1 = steady_clock::now();
    v = { s1,{} };
    stabfull = stabmin = change = 0;
    for (i = 1;; i = !i) {
        t2 = steady_clock::now();
        v[i] = S();
        change += SIDE(v[i]) != SIDE(v[!i]);
        stabmin += STABMIN(v[!i], v[i]);
        stabfull += STABFULL(v[!i], v[i]);
        roll = s1.empty() || change || v[i].empty();
        if (DIF(v[!i], v[i]))t1 = t2;
        b = BOARD(v[i]);
        ADD(b1, b, bs);
        if (roll || DICENEXT(s1, v[i]))s2 = v[i];
        if (roll || bs.size() && b == bs[0].board)b2 = b;
        if (v[i].empty() && s1.size() || roll == 0 && stabmin || roll && stabfull && (t2 - t1).count() >= 300000000)return;
    }
}
void LOAD() {
    int roll, change, side, from, to, piece;
    vector<int> s1, s2, b1, b2, way;
    for (;; END(s1, s2, b1, b2)) {
        NEW(roll, change, s1, s2, b1, b2);
        lock_guard<mutex> lock(posMutex);
        ROLL = roll;
        if (s2.empty()) {
            START(POS);
            continue;
        }
        BOARDSET(b2);
        POS.side = SIDE(s2);
        CASTLESET();
        if (s1.empty()) {
            DICESET(s2);
            POS.key = computeKey(POS);
            continue;
        }
        side = SIDE(s1);
        way = WAY(b1, b2);
        if (way.size() == 4) {
            POS.castle &= 12 - 9 * side;
            POS.dice = newDice[POS.dice][5];
            POS.dice = newDice[POS.dice][3];
        }
        else if (way.size() == 3)POS.dice = newDice[POS.dice][0]; else if (way.size() == 2) {
            from = way[0];
            to = way[1];
            piece = b1[from] % 6;
            if (piece == 0 && (from ^ to) == 16)POS.ep1[!side] |= bit((from + to) / 2);
            if (piece == 0 && POS.ep1[side] & epMask[to])POS.ep2 |= bit(to);
            POS.castle &= ~(MASK[from] | MASK[to]);
            POS.dice = newDice[POS.dice][piece];
        }
        if (change) {
            POS.ep1[side] = 0;
            POS.ep2 = 0;
            DICESET(s2);
        }
        if (change >= 2)POS.ep1[!side] = 0;
        POS.key = computeKey(POS);
    }
}
void SEARCH() {
    int clear, ready;
    float eval, depth;
    vector<int> pv;
    vector<moveState> moves;
    Position pos;
    MCTSTable T(1 << 23, 1 << 26);
    START(pos);
    while (1) {
        Sleep(1);
        ready = clear = 0;
        {
            lock_guard<mutex> lock(posMutex);
            if (POS.key != pos.key) {
                pos = POS;
                clear = ROLL;
                ready = POS.dice && POS.color[0] & POS.piece[5] && POS.color[1] & POS.piece[5];
            }
        }
        if (clear)T.newGame();
        if (ready)mctsBatchedMT(T, pos, PATH, MASK, INT_MAX, eval, depth, moves, pv, 2, 1, autoSearchThreads(), true);
    }
}

static bool isChanceOrTerminalPosition(Position& pos,
    const std::array<uint64_t, 4>& path,
    const std::array<int, 64>& mask,
    bool& outTerminal,
    bool& outChance) {
    MoveList ml;
    int term = 0;
    Position tmp = pos;
    genLegal(tmp, path, mask, ml, term);
    outTerminal = (term != 0);
    outChance = (!outTerminal && ml.n == 0);
    return outTerminal || outChance;
}

static bool analyzeAndPlayFirstPvMove(MCTSTable& T,
    Position& pos,
    std::array<uint64_t, 4>& path,
    std::array<int, 64>& mask,
    double seconds) {
    T.newGame();
    float eval = 0.5f;
    float depth = 0.0f;
    std::vector<moveState> moves;
    std::vector<int> pv;
    mctsBatchedMT(T, pos, path, mask, seconds, eval, depth, moves, pv, 0, 0);
    if (pv.empty()) return false;
    makeMove(pos, mask, pv[0]);
    return true;
}

static bool analyzeOnceAndPlayPvUntilChance(MCTSTable& T,
    Position& pos,
    std::array<uint64_t, 4>& path,
    std::array<int, 64>& mask,
    double seconds) {
    T.newGame();
    float eval = 0.5f;
    float depth = 0.0f;
    std::vector<moveState> moves;
    std::vector<int> pv;
    mctsBatchedMT(T, pos, path, mask, seconds, eval, depth, moves, pv, 0, 0);
    if (pv.empty()) return false;

    for (int mv : pv) {
        bool terminal = false;
        bool chance = false;
        if (isChanceOrTerminalPosition(pos, path, mask, terminal, chance)) return true;
        makeMove(pos, mask, mv);
    }
    return true;
}

static int playOneTimedPvMatchGame(const Position& startPos,
    std::array<uint64_t, 4> path,
    std::array<int, 64> mask,
    bool p1IsWhite,
    const std::vector<int>* mirroredDice,
    std::vector<int>* producedDice,
    int maxPlies = 512) {
    Position pos = startPos;
    MCTSTable p1Table(1ull << 23, 1ull << 26);
    MCTSTable p2Table(1ull << 23, 1ull << 26);
    size_t chanceIdx = 0;

    for (int ply = 0; ply < maxPlies; ++ply) {
        bool terminal = false;
        bool chance = false;
        isChanceOrTerminalPosition(pos, path, mask, terminal, chance);

        if (terminal) {
            const bool p1Won = ((pos.side == 0) == p1IsWhite);
            return p1Won ? +1 : -1;
        }

        if (chance) {
            TTNode* n = (((pos.side == 0) == p1IsWhite) ? p1Table : p2Table).findNodeNoInsert(pos.key);
            if (mirroredDice && chanceIdx < mirroredDice->size()) {
                makeRandomWithRolledDice(pos, n, (*mirroredDice)[chanceIdx]);
            }
            else {
                const int rolledDice = Dice[Range(Random)];
                if (producedDice) producedDice->push_back(rolledDice);
                makeRandomWithRolledDice(pos, n, rolledDice);
            }
            ++chanceIdx;
            continue;
        }

        const bool p1Turn = ((pos.side == 0) == p1IsWhite);
        if (p1Turn) {
            do {
                if (!analyzeAndPlayFirstPvMove(p1Table, pos, path, mask, 1.0)) return 0;
                isChanceOrTerminalPosition(pos, path, mask, terminal, chance);
            } while (!terminal && !chance);
        }
        else {
            if (!analyzeOnceAndPlayPvUntilChance(p2Table, pos, path, mask, 3.0)) return 0;
        }
    }

    return 0;
}

static MatchStatsGeneric runTimedPvMatch(int games) {
    MatchStatsGeneric st;
    for (int g = 0; g < games; g += 2) {
        Position startPos;
        std::array<uint64_t, 4> path;
        std::array<int, 64> mask;
        chess960(startPos, path, mask);

        std::vector<int> firstGameDice;
        int r1 = playOneTimedPvMatchGame(startPos, path, mask, true, nullptr, &firstGameDice);
        if (r1 > 0) ++st.p1Wins; else if (r1 < 0) ++st.p2Wins; else ++st.draws;

        if (g + 1 < games) {
            int r2 = playOneTimedPvMatchGame(startPos, path, mask, false, &firstGameDice, nullptr);
            if (r2 > 0) ++st.p1Wins; else if (r2 < 0) ++st.p2Wins; else ++st.draws;
        }

        const int played = std::min(g + 2, games);
        if ((played % 10) == 0 || played == games) {
            std::cout << "[timed-pv-match] games=" << played
                << " p1/p2/draw=" << st.p1Wins << '/' << st.p2Wins << '/' << st.draws
                << " p1Score=" << std::fixed << std::setprecision(4) << st.p1Score() << '\n';
        }
    }
    return st;
}

// ===================== A/B strength match: final vs original =====================
// P1 = "final": multi-threaded leaf production + tree reuse across the game.
// P2 = "original": exact old play behavior (1 thread, fresh tree before every move).
// Paired games: same Chess960 opening, same dice sequence, colors swapped.

struct AbPlayerCfg {
    unsigned threads = 1;
    bool treeReuse = false;
    bool dualInfer = false;
    bool oldNet = false;        // use g_trt_old (net_old.plan) instead of g_trt
    bool persistServer = false; // keep one InferenceServer per game instead of per move
    double moveTimeSec = 0.4;
    int adaptive = 0;           // 0 = fixed, 1 = settled leader, 2 = final-position gap
    double bankSec = 0.0;       // >0: a per-game clock both sides must live within
    const char* name = "";

    TrtRunner* primary() const {
        return (oldNet && g_trtOldReady) ? &g_trt_old : nullptr;
    }
};

// Time actually consumed by each side, so a match can show that the adaptive
// player won on the same clock rather than on a bigger one.
static double g_abTimeP1 = 0.0, g_abTimeP2 = 0.0;
static long long g_abMovesP1 = 0, g_abMovesP2 = 0;
static long long g_abTurnsP1 = 0, g_abTurnsP2 = 0;   // full turns, i.e. dice rolls

static void abPrepareTableForSearch(MCTSTable& T, const AbPlayerCfg& cfg) {
    if (!cfg.treeReuse) { T.newGame(); return; }
    // Reuse mode: keep the tree, but reset if the edge pool is close to full
    // or a previous search aborted on overflow.
    const bool aborted = T.abort.load(std::memory_order_relaxed);
    const double edgeFill =
        (double)T.edgeTop.load(std::memory_order_relaxed) / (double)T.edges.size();
    if (aborted || edgeFill > 0.75) T.newGame();
}

static int playOneAbMatchGame(const AbPlayerCfg& p1Cfg,
    const AbPlayerCfg& p2Cfg,
    MCTSTable& p1Table,
    MCTSTable& p2Table,
    const Position& startPos,
    std::array<uint64_t, 4> path,
    std::array<int, 64> mask,
    bool p1IsWhite,
    const std::vector<int>* mirroredDice,
    std::vector<int>* producedDice,
    int maxPlies = 512) {
    Position pos = startPos;
    p1Table.newGame();
    p2Table.newGame();
    size_t chanceIdx = 0;
    double bankP1 = 0.0, bankP2 = 0.0;    // time saved so far, spendable later
    std::vector<int> lineP1, lineP2;      // series still to be played out

    // Optional per-game persistent inference servers (skip per-move start/stop/ramp costs).
    struct SrvHolder {
        std::unique_ptr<InferenceServer> s;
        ~SrvHolder() noexcept { if (s) { try { s->stopAndDrain(); } catch (...) {} } }
    };
    SrvHolder p1Srv, p2Srv;
    if (p1Cfg.persistServer) {
        TrtRunner* rt = p1Cfg.primary();
        const bool dual = p1Cfg.dualInfer && g_trt2Ready && !rt;
        p1Srv.s = std::make_unique<InferenceServer>(p1Table, rt ? rt : &g_trt, dual ? &g_trt2 : nullptr);
        p1Srv.s->start();
    }
    if (p2Cfg.persistServer) {
        TrtRunner* rt = p2Cfg.primary();
        const bool dual = p2Cfg.dualInfer && g_trt2Ready && !rt;
        p2Srv.s = std::make_unique<InferenceServer>(p2Table, rt ? rt : &g_trt, dual ? &g_trt2 : nullptr);
        p2Srv.s->start();
    }

    for (int ply = 0; ply < maxPlies; ++ply) {
        bool terminal = false;
        bool chance = false;
        isChanceOrTerminalPosition(pos, path, mask, terminal, chance);

        if (terminal) {
            const bool p1Won = ((pos.side == 0) == p1IsWhite);
            return p1Won ? +1 : -1;
        }

        if (chance) {
            TTNode* n = (((pos.side == 0) == p1IsWhite) ? p1Table : p2Table).findNodeNoInsert(pos.key);
            if (mirroredDice && chanceIdx < mirroredDice->size()) {
                makeRandomWithRolledDice(pos, n, (*mirroredDice)[chanceIdx]);
            }
            else {
                const int rolledDice = Dice[Range(Random)];
                if (producedDice) producedDice->push_back(rolledDice);
                makeRandomWithRolledDice(pos, n, rolledDice);
            }
            ++chanceIdx;
            // A fresh roll starts a new turn for whoever is now to move, and
            // the tree from the previous roll is of no use to it.
            if ((pos.side == 0) == p1IsWhite) { lineP1.clear(); ++g_abTurnsP1; p1Table.newGame(); }
            else { lineP2.clear(); ++g_abTurnsP2; p2Table.newGame(); }
            continue;
        }

        const bool p1Turn = ((pos.side == 0) == p1IsWhite);
        const AbPlayerCfg& cfg = p1Turn ? p1Cfg : p2Cfg;
        MCTSTable& T = p1Turn ? p1Table : p2Table;

        abPrepareTableForSearch(T, cfg);

        // One search per turn: the whole series of moves is taken from it, so
        // the question being tested is how to spend a turn's allowance, not a
        // position's. Later moves of the series are replayed from that line.
        std::vector<int>& line = p1Turn ? lineP1 : lineP2;
        if (!line.empty()) {
            const int mv = line.front();
            line.erase(line.begin());
            MoveList ml; int t = 0;
            { Position probe = pos; genLegal(probe, path, mask, ml, t); }
            bool legal = false;
            for (int i = 0; i < ml.n; ++i) if (ml.m[i] == mv) { legal = true; break; }
            if (legal) { makeMove(pos, mask, mv); continue; }
            line.clear();                    // stale line, search again
        }

        double& bank = p1Turn ? bankP1 : bankP2;
        // The adaptive side gets a ceiling, not an allowance: how much of it is
        // actually used is decided per position, and the average is pulled back
        // to the nominal figure by the controller below. Tying the ceiling to
        // saved-up time meant a hard position could only be studied after a
        // string of easy ones, which is not what the rule is for.
        double limit = cfg.adaptive ? cfg.moveTimeSec * 3.0 : cfg.moveTimeSec;
        if (!cfg.adaptive) bank = 0.0;
        if (limit < 0.02) limit = 0.02;
        if (p1Turn) ++g_abMovesP1; else ++g_abMovesP2;

        float eval = 0.5f;
        float depth = 0.0f;
        std::vector<moveState> moves;
        std::vector<int> pv;
        const auto tMove = std::chrono::steady_clock::now();
        mctsBatchedMT(T, pos, path, mask, limit, eval, depth, moves, pv,
            /*write=*/0, /*abort=*/0, cfg.threads, cfg.dualInfer,
            cfg.primary(), (p1Turn ? p1Srv : p2Srv).s.get(), /*stopOnWin=*/true,
            cfg.adaptive, cfg.moveTimeSec, nullptr,
            cfg.adaptive ? (p1Turn ? g_creditP1 : g_creditP2) : 0.0);
        const double used =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - tMove).count();
        // Savings are measured against the nominal allowance, not against the
        // raised limit: crediting the limit feeds itself and the average drifts
        // upwards, which would quietly hand the adaptive side more time.
        bank += cfg.moveTimeSec - used;
        if (cfg.adaptive && cfg.moveTimeSec > 0.0) {
            double& cr = p1Turn ? g_creditP1 : g_creditP2;
            const double ratio = std::max(0.2, used / cfg.moveTimeSec);
            // Aim at what the fixed side actually spends, not at the nominal
            // figure: starting and finishing a search costs a few percent that
            // no policy can give back, and chasing the nominal would leave the
            // adaptive side permanently over budget.
            // Compare the running averages, not this one move: the effect being
            // measured is a few Elo, and a few percent of budget imbalance is
            // worth about as much, so the two sides have to be matched to well
            // under a percent or the experiment measures its own error.
            const long long om = p1Turn ? g_abMovesP2 : g_abMovesP1;
            const double ot = p1Turn ? g_abTimeP2 : g_abTimeP1;
            const long long mm = p1Turn ? g_abMovesP1 : g_abMovesP2;
            const double mt = p1Turn ? g_abTimeP1 : g_abTimeP2;
            if (om > 200 && mm > 200) {
                const double theirs = ot / om;
                const double mine = mt / mm;
                cr *= std::pow(theirs / std::max(1e-6, mine), 0.02);
                cr = std::min(3.0, std::max(0.3, cr));
            }
            (void)ratio;
        }
        if (p1Turn) g_abTimeP1 += used; else g_abTimeP2 += used;
        if (pv.empty()) return 0;

        // The series of most-visited moves, all the way to the next roll.
        uint64_t endKey = 0;
        line.clear();
        extractBestPVUntilChance(T, pos, mask, line, endKey);
        if (line.empty()) line.push_back(pv[0]);
        const int first = line.front();
        line.erase(line.begin());
        makeMove(pos, mask, first);
    }

    return 0;
}

static double abEloFromScore(double s) {
    if (s <= 0.0) return -999.0;
    if (s >= 1.0) return 999.0;
    return -400.0 * std::log10(1.0 / s - 1.0);
}

static MatchStatsGeneric runAbStrengthMatch(int games, double moveTimeSec,
    AbPlayerCfg p1Cfg, AbPlayerCfg p2Cfg) {
    p1Cfg.moveTimeSec = moveTimeSec;
    // bankSec doubles as "give P2 a different allowance", which turns the same
    // harness into a sanity check: does more time buy strength at all here?
    p2Cfg.moveTimeSec = p2Cfg.bankSec > 0.0 ? p2Cfg.bankSec : moveTimeSec;

    auto printCfg = [](const char* tag, const AbPlayerCfg& c) {
        std::cout << tag << ": threads=" << c.threads
            << " reuse=" << (int)c.treeReuse
            << " dual=" << (int)c.dualInfer
            << " oldNet=" << (int)c.oldNet
            << " persist=" << (int)c.persistServer
            << " adaptive=" << (int)c.adaptive
            << " bank=" << c.bankSec << "s";
        };
    std::cout << "[ab] ";
    printCfg("P1", p1Cfg);
    std::cout << " | ";
    printCfg("P2", p2Cfg);
    std::cout << " | moveTime=" << moveTimeSec << "s games=" << games << std::endl;

    size_t n1, e1, n2, e2;
    tableSizeForTime(p1Cfg.moveTimeSec * 3.0, n1, e1);   // adaptive may triple a turn
    tableSizeForTime(p2Cfg.moveTimeSec * 3.0, n2, e2);
    std::cout << "[ab] tree P1=" << n1 << "n/" << e1 << "e P2=" << n2 << "n/" << e2 << "e\n";
    MCTSTable p1Table(n1, e1);
    MCTSTable p2Table(n2, e2);

    MatchStatsGeneric st;
    for (int g = 0; g < games; g += 2) {
        Position startPos;
        std::array<uint64_t, 4> path;
        std::array<int, 64> mask;
        chess960(startPos, path, mask);

        std::vector<int> firstGameDice;
        int r1 = playOneAbMatchGame(p1Cfg, p2Cfg, p1Table, p2Table,
            startPos, path, mask, /*p1IsWhite=*/true, nullptr, &firstGameDice);
        if (r1 > 0) ++st.p1Wins; else if (r1 < 0) ++st.p2Wins; else ++st.draws;

        if (g + 1 < games) {
            int r2 = playOneAbMatchGame(p1Cfg, p2Cfg, p1Table, p2Table,
                startPos, path, mask, /*p1IsWhite=*/false, &firstGameDice, nullptr);
            if (r2 > 0) ++st.p1Wins; else if (r2 < 0) ++st.p2Wins; else ++st.draws;
        }

        const int played = std::min(g + 2, games);
        const int n = st.p1Wins + st.p2Wins + st.draws;
        const double score = n ? (st.p1Wins + 0.5 * st.draws) / (double)n : 0.5;
        std::cout << "[ab] games=" << played
            << " W/L/D=" << st.p1Wins << '/' << st.p2Wins << '/' << st.draws
            << " score=" << std::fixed << std::setprecision(4) << score
            << " elo=" << std::setprecision(1) << std::showpos << abEloFromScore(score) << std::noshowpos
            << " LOS=" << std::setprecision(2) << computeLOSPercent(st.p1Wins, st.p2Wins) << '%'
            << std::setprecision(3)
            << " time P1/P2=" << g_abTimeP1 << "s/" << g_abTimeP2 << "s"
            << " perTurn=" << (g_abTurnsP1 ? g_abTimeP1 / g_abTurnsP1 : 0.0)
            << "/" << (g_abTurnsP2 ? g_abTimeP2 / g_abTurnsP2 : 0.0)
            << " searchesPerTurn=" << (g_abTurnsP1 ? (double)g_abMovesP1 / g_abTurnsP1 : 0.0)
            << " earlyStops=" << g_adaptStopSettled.load() << "settled/"
            << g_adaptStopFree.load() << "free of " << g_adaptSearches.load()
            << " gap=" << (g_gapN ? g_gapSum / g_gapN : 0.0)
            << [] {
                if (g_spreadN < 500) return std::string();
                std::vector<double> v(g_spreadSamples.begin(), g_spreadSamples.begin() + g_spreadN);
                std::sort(v.begin(), v.end());
                auto q = [&](double p) { return v[(size_t)(p * (v.size() - 1))]; };
                std::ostringstream os;
                os << std::fixed << std::setprecision(4)
                    << " importance p10/p50/p90=" << q(0.10) << '/' << q(0.50) << '/' << q(0.90)
                    << " ratio=" << std::setprecision(1) << (q(0.90) / std::max(1e-6, q(0.10)));
                return os.str();
            }()
            << " probe=" << (g_probeN ? g_probeSec / g_probeN * 1e6 : 0.0) << "us x"
            << g_probeN << " (" << std::setprecision(4)
            << (g_abTimeP1 > 0.0 ? 100.0 * g_probeSec / g_abTimeP1 : 0.0) << "%)"
            << std::setprecision(3)
            << std::setprecision(6) << std::endl;
    }
    return st;
}

// Measures what the "dif" of the best root move actually looks like in play:
// its sign, its scale, and how often a move turns out to cut nothing off. The
// time policy below is built on those numbers, so they are worth measuring
// rather than guessing.
static void difProbe(int positions, double seconds) {
    MCTSTable T(1ull << 22, 1ull << 25);
    Position pos;
    std::array<uint64_t, 4> path;
    std::array<int, 64> mask;
    chess960(pos, path, mask);
    int seen = 0, freeMoves = 0;
    std::vector<double> difs;

    for (int i = 0; seen < positions && i < positions * 40; ++i) {
        bool terminal = false, chance = false;
        isChanceOrTerminalPosition(pos, path, mask, terminal, chance);
        if (terminal) { chess960(pos, path, mask); T.newGame(); continue; }
        if (chance) { makeRandomWithRolledDice(pos, T.findNodeNoInsert(pos.key), Dice[Range(Random)]); continue; }

        MoveList ml; int term = 0;
        Position probe = pos;
        genLegal(probe, path, mask, ml, term);

        float eval = 0.5f, depth = 0.0f;
        std::vector<moveState> rm;
        std::vector<int> pv;
        mctsBatchedMT(T, pos, path, mask, seconds, eval, depth, rm, pv, 0, 0,
            autoSearchThreads(), true);
        if (pv.empty()) { chess960(pos, path, mask); T.newGame(); continue; }

        if (ml.n >= 2 && !rm.empty()) {
            ++seen;
            const moveState* best = &rm[0];
            for (const moveState& ms : rm) if (ms.move == pv[0]) { best = &ms; break; }
            difs.push_back(best->dif);
            if (best->dif >= 99.0) ++freeMoves;
            std::cout << "[dif] moves=" << ml.n
                << " best=" << moveToStr(pv[0])
                << " dif=" << std::fixed << std::setprecision(2) << best->dif
                << "  all:";
            int shown = 0;
            for (const moveState& ms : rm) {
                if (shown++ >= 5) break;
                std::cout << ' ' << moveToStr(ms.move) << '/' << ms.visits << '/' << ms.dif;
            }
            std::cout << '\n';
        }
        makeMove(pos, mask, pv[0]);
    }

    std::sort(difs.begin(), difs.end());
    std::cout << "[dif] positions=" << difs.size()
        << " cutsNothing=" << freeMoves;
    if (!difs.empty()) {
        auto q = [&](double p) { return difs[(size_t)(p * (difs.size() - 1))]; };
        std::cout << " min=" << q(0) << " p25=" << q(0.25) << " median=" << q(0.5)
            << " p75=" << q(0.75) << " max=" << q(1);
    }
    std::cout << std::endl;
}

// ===================== ENGINE PLAY ON SCREEN (dicechess.com) =====================
// Reads the board, the dice and whose turn it is straight from the browser
// window, searches with the net and plays the moves with the mouse.
//
// calib.txt    "boardX boardY cellSize": physical pixels of the board's top-left
//              corner and the size of one square. Everything else is derived
//              from it and scaled by cellSize/139.
// boardcal.txt learned piece shapes, rewritten automatically every time a game
//              starts from the initial position.

namespace SP {

    static int BX = 404, BY = 757, CELL = 139;

    static double sc() { return CELL / 139.0; }
    static int    rel(double v) { return (int)llround(v * sc()); }
    static int    boardW() { return 8 * CELL; }
    static int    cellX(int f) { return BX + CELL * f; }
    static int    cellY(int r) { return BY + CELL * r; }
    static int    dieSize() { return rel(159); }
    static int    dieX(int i) { return BX + boardW() / 2 + (i - 1) * rel(228) - dieSize() / 2; }
    static int    dieY() { return BY - rel(212); }
    static int    promoY() { return BY + boardW() + rel(84); }
    static int    promoX(int i) { return BX + boardW() / 2 + (int)llround((i - 1.5) * rel(182)); }
    static int    rematchX() { return BX + rel(318); }
    static int    rematchY() { return BY + rel(1185); }

    // ---------------------------------------------------------------- capture

    struct Shot {
        int x0 = 0, y0 = 0, w = 0, h = 0;
        vector<uint32_t> px;
        bool ok() const { return w > 0 && h > 0; }
        // Outside the capture returns magenta, i.e. never "ink" and never "lit".
        inline uint32_t at(int x, int y) const {
            x -= x0; y -= y0;
            if (x < 0 || y < 0 || x >= w || y >= h) return 0x00FF00FFu;
            return px[(size_t)y * w + x];
        }
    };

    static Shot grab(int x0, int y0, int x1, int y1) {
        Shot s;
        s.x0 = x0; s.y0 = y0; s.w = x1 - x0; s.h = y1 - y0;
        if (s.w <= 0 || s.h <= 0) { s.w = s.h = 0; return s; }
        s.px.resize((size_t)s.w * s.h);
        HDC d = GetDC(0);
        HDC m = CreateCompatibleDC(d);
        HBITMAP b = CreateCompatibleBitmap(d, s.w, s.h);
        BITMAPINFO bi{};
        bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
        bi.bmiHeader.biWidth = s.w;
        bi.bmiHeader.biHeight = -s.h;          // top-down
        bi.bmiHeader.biPlanes = 1;
        bi.bmiHeader.biBitCount = 32;
        bi.bmiHeader.biCompression = BI_RGB;
        HGDIOBJ old = SelectObject(m, b);
        BitBlt(m, 0, 0, s.w, s.h, d, x0, y0, SRCCOPY);
        GetDIBits(d, b, 0, s.h, s.px.data(), &bi, DIB_RGB_COLORS);
        SelectObject(m, old);
        DeleteObject(b);
        DeleteDC(m);
        ReleaseDC(0, d);
        return s;
    }

    // One capture holding the board, the dice, both clocks, the promotion row
    // and the end-of-game buttons.
    static Shot grabAll() {
        return grab(BX - rel(90), BY - rel(470),
            BX + boardW() + rel(250), BY + boardW() + rel(215));
    }

    // ------------------------------------------------------------- descriptor
    // A piece is recognised by the SHAPE of its ink, not by pixel colours: the
    // drawings on the dice are the drawings from the board, only smaller, so a
    // bbox-normalised 6x6 occupancy grid matches both.

    // g      = how much of each 6x6 bbox cell is covered by ink (per mille)
    // centre = brightness of the ink in the middle of the drawing, rescaled to
    //          the patch's own ink range. The middle is always fill, so this
    //          separates a white piece from a black one, and the rescaling
    //          keeps it working on a dimmed die where white is painted ~120.
    struct Desc {
        array<int, 36> g{};
        int aspect = 0;        // 1000*w/h of the ink bbox
        int ink = 0;
        int centre = 128;
    };

    // "Ink" = unsaturated pixel. Wood, felt and the red dice faces are strongly
    // saturated; the drawings are white/grey/black. Exact colour matching does
    // not work here because a bright piece is painted (254,254,254), not white.
    static AI_FORCEINLINE bool isInk(uint32_t v, int& lum) {
        int B = (int)(v & 255u), G = (int)((v >> 8) & 255u), R = (int)((v >> 16) & 255u);
        int mx = R > G ? (R > B ? R : B) : (G > B ? G : B);
        int mn = R < G ? (R < B ? R : B) : (G < B ? G : B);
        lum = mx;
        return mx - mn <= 28;
    }

    static Desc describe(const Shot& s, int x0, int y0, int size, int inset) {
        Desc d;
        if (size <= 2 * inset + 4) return d;
        const int n = size;
        static thread_local vector<unsigned char> m;
        static thread_local vector<unsigned char> lu;
        m.assign((size_t)n * n, 0);
        lu.assign((size_t)n * n, 0);
        int hist[256] = { 0 };
        int bx0 = 1 << 30, by0 = 1 << 30, bx1 = -1, by1 = -1;
        for (int y = inset; y < n - inset; ++y) {
            for (int x = inset; x < n - inset; ++x) {
                int lum;
                if (!isInk(s.at(x0 + x, y0 + y), lum)) continue;
                m[(size_t)y * n + x] = 1;
                lu[(size_t)y * n + x] = (unsigned char)lum;
            }
        }
        // Keep only the large connected blobs. The rank digits and file letters
        // printed in the corners of a square are ink too, and left in they drag
        // the bounding box sideways, which is enough to turn a rook into
        // something the learned table does not recognise.
        {
            static thread_local vector<int> comp, stack, sizes;
            comp.assign((size_t)n * n, -1);
            sizes.clear();
            for (int y = 0; y < n; ++y) for (int x = 0; x < n; ++x) {
                size_t p = (size_t)y * n + x;
                if (!m[p] || comp[p] >= 0) continue;
                int id = (int)sizes.size();
                int cnt = 0;
                stack.clear();
                stack.push_back((int)p);
                comp[p] = id;
                while (!stack.empty()) {
                    int q = stack.back(); stack.pop_back();
                    ++cnt;
                    int qy = q / n, qx = q % n;
                    const int dx[4] = { 1,-1,0,0 }, dy[4] = { 0,0,1,-1 };
                    for (int k = 0; k < 4; ++k) {
                        int nx = qx + dx[k], ny = qy + dy[k];
                        if (nx < 0 || ny < 0 || nx >= n || ny >= n) continue;
                        size_t r = (size_t)ny * n + nx;
                        if (!m[r] || comp[r] >= 0) continue;
                        comp[r] = id;
                        stack.push_back((int)r);
                    }
                }
                sizes.push_back(cnt);
            }
            int big = 0;
            for (int v : sizes) big = max(big, v);
            int keep = max(30, big / 8);
            for (size_t p = 0; p < m.size(); ++p)
                if (m[p] && sizes[comp[p]] < keep) m[p] = 0;
        }
        for (int y = 0; y < n; ++y) for (int x = 0; x < n; ++x) {
            if (!m[(size_t)y * n + x]) continue;
            d.ink++;
            hist[lu[(size_t)y * n + x]]++;
            if (x < bx0) bx0 = x;
            if (x > bx1) bx1 = x;
            if (y < by0) by0 = y;
            if (y > by1) by1 = y;
        }
        if (d.ink == 0) return d;
        // Brightness is rescaled to the ink's own range: on a dimmed die a
        // white piece is painted around 120, so absolute levels are useless.
        int lo = 0, hi = 255, acc = 0;
        for (int v = 0; v < 256; ++v) { acc += hist[v]; if (acc >= d.ink / 20 + 1) { lo = v; break; } }
        acc = 0;
        for (int v = 255; v >= 0; --v) { acc += hist[v]; if (acc >= d.ink / 20 + 1) { hi = v; break; } }
        if (hi <= lo) hi = lo + 1;

        int w = bx1 - bx0 + 1, h = by1 - by0 + 1;
        d.aspect = 1000 * w / h;
        array<int, 36> cnt{}, tot{};
        array<long long, 36> lsum{};
        for (int y = by0; y <= by1; ++y) {
            int gy = (y - by0) * 6 / h;
            for (int x = bx0; x <= bx1; ++x) {
                int k = gy * 6 + (x - bx0) * 6 / w;
                tot[k]++;
                if (!m[(size_t)y * n + x]) continue;
                cnt[k]++;
                int v = ((int)lu[(size_t)y * n + x] - lo) * 255 / (hi - lo);
                lsum[k] += min(255, max(0, v));
            }
        }
        for (int k = 0; k < 36; ++k) d.g[k] = tot[k] ? 1000 * cnt[k] / tot[k] : 0;
        {
            long long ls = 0;
            int lc = 0;
            for (int k : {14, 15, 20, 21}) { ls += lsum[k]; lc += cnt[k]; }
            if (lc) d.centre = (int)(ls / lc);
        }
        return d;
    }

    // Absolute brightness of the ink in the middle of a square, used only for
    // the very first orientation guess, before anything has been learned.
    static int centreLum(const Shot& s, int cell) {
        int side = max(8, CELL / 4);
        int x0 = cellX(cell & 7) + CELL / 2 - side / 2;
        int y0 = cellY(cell >> 3) + CELL / 2 - side / 2;
        long long sum = 0;
        int n = 0;
        for (int y = 0; y < side; ++y) for (int x = 0; x < side; ++x) {
            int lum;
            if (!isInk(s.at(x0 + x, y0 + y), lum)) continue;
            sum += lum; n++;
        }
        return n ? (int)(sum / n) : -1;
    }

    // Silhouette plus one brightness number. The drawings on the dice carry a
    // thicker outline than the ones on the board, so per-cell brightness does
    // not carry over, but the outline grows outwards and the bbox-normalised
    // shape still matches. The centre brightness is compared against the
    // learned value rather than a fixed threshold: on a knight the middle of
    // the bounding box lands on the mane, so "black piece = dark centre" is
    // simply not true, while "black knight = the centre of a black knight" is.
    static int descDist(const Desc& a, const Desc& b) {
        int s = 0;
        for (int i = 0; i < 36; ++i) s += abs(a.g[i] - b.g[i]);
        s += abs(a.aspect - b.aspect) / 2;
        s += 8 * abs(a.centre - b.centre);
        return s;
    }

    // ------------------------------------------------------------ calibration

    // Several prototypes per class rather than one average: the two knights of
    // a colour are drawn on squares of different shade and averaging them
    // produces a shape that matches neither.
    static const int MAXPROTO = 4;
    struct Cal {
        Desc proto[2][6][MAXPROTO];   // [0] = white drawings, [1] = black drawings
        int  n[2][6] = { {0} };
        int  have = 0;
        int  emptyInk = 0;            // largest ink count seen on a known empty square
    };
    static Cal CAL;

    static int classDist(const Desc& d, int colour, int type);

    static int inkEmptyLimit() { return max(400, CAL.emptyInk * 3); }

    // Prints the current shapes in the format of kBoardCal below, for pasting
    // back into the source if the site ever changes how it draws the pieces.
    // Nothing is written to disk: results.txt is the only file this mode keeps.
    static void dumpCal() {
        cout << CAL.emptyInk << '\n';
        for (int c = 0; c < 2; ++c) for (int t = 0; t < 6; ++t) {
            cout << c << ' ' << t << ' ' << CAL.n[c][t] << '\n';
            for (int k = 0; k < CAL.n[c][t]; ++k) {
                const Desc& d = CAL.proto[c][t][k];
                cout << d.aspect << ' ' << d.ink << ' ' << d.centre;
                for (int i = 0; i < 36; ++i) cout << ' ' << d.g[i];
                cout << '\n';
            }
        }
    }

// Piece shapes learned from the initial position, baked into the binary so
// the engine needs no calibration file. They are still relearned whenever a
// game starts from the initial position, which keeps them in step with any
// change to how the site draws the pieces.
static const char* const kBoardCal =
    "0 0 0 1 778 4247 204 0 282 900 883 226 0 0 610 1000 1000 530 0 0 "
    "704 1000 1000 629 0 0 352 1000 990 281 0 360 961 1000 1000 941 329 954 1000 1000 "
    "1000 1000 953 0 1 2 926 7754 212 9 668 767 139 0 0 98 960 1000 1000 732 "
    "38 457 1000 1000 1000 1000 500 911 996 937 1000 1000 805 526 581 937 1000 1000 961 0 "
    "509 964 1000 993 944 926 7714 213 15 659 761 114 0 0 101 967 1000 1000 722 27 "
    "464 1000 1000 1000 1000 468 918 996 934 1000 1000 784 522 565 941 1000 1000 930 0 526 "
    "957 1000 986 940 0 2 2 990 5244 191 0 0 470 473 0 0 0 182 913 922 "
    "185 0 0 614 1000 1000 629 0 0 373 1000 1000 382 0 0 373 1000 1000 379 0 "
    "570 722 817 817 722 580 981 5277 192 0 0 453 450 0 0 0 166 891 891 160 "
    "0 0 623 1000 1000 623 0 0 388 1000 1000 388 0 0 385 1000 1000 376 0 595 "
    "722 827 824 722 589 0 3 2 867 6964 237 607 807 870 843 825 503 229 911 1000 "
    "1000 870 185 0 733 1000 1000 662 0 0 762 1000 1000 677 0 281 1000 1000 1000 996 "
    "225 919 1000 1000 1000 1000 874 858 6967 237 611 781 874 829 822 585 236 933 1000 1000 "
    "885 211 0 788 1000 1000 666 0 0 814 1000 1000 696 0 329 1000 1000 1000 1000 274 "
    "926 1000 1000 1000 1000 917 0 4 1 1091 7227 136 150 500 360 326 465 149 608 402 "
    "300 266 386 634 400 713 775 772 716 353 247 1000 1000 1000 1000 195 2 833 1000 1000 "
    "772 0 0 883 1000 994 822 0 0 5 1 990 6818 164 0 0 254 204 0 0 "
    "99 268 620 574 274 95 871 1000 1000 1000 1000 885 725 1000 1000 1000 1000 728 35 916 "
    "1000 1000 873 24 0 638 966 962 595 0 1 0 1 778 3960 11 0 262 914 896 "
    "208 0 0 480 1000 1000 403 0 0 553 1000 990 487 0 0 152 959 921 116 0 "
    "256 888 1000 1000 858 226 918 1000 1000 1000 1000 918 1 1 2 936 7575 10 11 547 "
    "656 74 0 0 104 964 1000 1000 653 13 450 1000 1000 1000 1000 388 903 944 814 1000 "
    "1000 715 441 421 970 1000 1000 885 0 581 944 950 944 885 865 7606 9 13 576 673 "
    "120 0 0 155 976 1000 1000 720 32 563 1000 1000 1000 1000 458 955 779 805 1000 1000 "
    "764 119 488 1000 1000 1000 926 0 325 473 479 473 492 1 2 2 990 5263 61 0 "
    "0 464 453 0 0 0 182 907 907 175 0 0 635 1000 1000 620 0 0 398 1000 "
    "1000 382 0 0 404 1000 1000 376 0 604 722 820 824 722 589 990 5203 61 0 0 "
    "470 438 0 0 0 185 907 895 166 0 0 632 1000 1000 601 0 0 401 1000 1000 "
    "367 0 0 398 1000 1000 358 0 586 719 811 805 700 561 1 3 2 855 6394 13 "
    "588 814 855 814 859 492 219 874 1000 1000 819 168 0 600 1000 1000 466 0 0 648 "
    "1000 1000 522 0 298 1000 1000 1000 992 193 866 941 945 941 972 848 862 6288 6 600 "
    "827 899 878 854 567 223 874 1000 1000 827 189 0 600 1000 1000 466 0 0 623 1000 "
    "1000 501 0 278 1000 1000 1000 992 205 886 1000 1000 1000 1000 878 1 4 1 1072 7267 "
    "36 205 550 351 347 505 216 580 413 330 330 361 611 425 752 836 844 769 374 215 "
    "997 1000 1000 992 155 0 838 1000 1000 763 0 11 750 953 947 688 0 1 5 1 "
    "990 6836 86 0 0 257 204 0 0 114 290 623 580 293 101 897 1000 1000 1000 1000 "
    "895 742 1000 1000 1000 1000 728 29 916 1000 1000 870 24 0 623 950 947 577 0 "
    ;

    static bool loadCal() {
        istringstream f(kBoardCal);
        int emptyInk = 0;
        if (!(f >> emptyInk)) return false;
        Cal c;
        c.emptyInk = emptyInk;
        for (int i = 0; i < 12; ++i) {
            int col, t, np;
            if (!(f >> col >> t >> np)) return false;
            if (col < 0 || col > 1 || t < 0 || t > 5 || np < 1 || np > MAXPROTO) return false;
            c.n[col][t] = np;
            for (int k = 0; k < np; ++k) {
                Desc& d = c.proto[col][t][k];
                if (!(f >> d.aspect >> d.ink >> d.centre)) return false;
                for (int j = 0; j < 36; ++j) if (!(f >> d.g[j])) return false;
            }
        }
        c.have = 1;
        CAL = c;
        return true;
    }

    static void loadGeometry(const char* file = "calib.txt") {
        ifstream f(file);
        int x, y, c;
        if (f >> x >> y >> c && c > 40) { BX = x; BY = y; CELL = c; }
    }

    // ------------------------------------------------------------ board & dice

    // Cell index on screen = row*8 + file, row 0 is the top row of the board.
    static Desc cellDesc(const Shot& s, int cell) {
        return describe(s, cellX(cell & 7), cellY(cell >> 3), CELL, rel(5));
    }
    static Desc diceDesc(const Shot& s, int i) {
        return describe(s, dieX(i), dieY(), dieSize(), rel(16));
    }

    static int classDist(const Desc& d, int colour, int type) {
        int best = INT_MAX;
        for (int k = 0; k < CAL.n[colour][type]; ++k)
            best = min(best, descDist(d, CAL.proto[colour][type][k]));
        return best;
    }

    // 0..5 white piece, 6..11 black piece, 12 empty, -1 unrecognised.
    static int classify(const Desc& d, int& miss, int& gap) {
        miss = 0; gap = 0;
        if (d.ink <= inkEmptyLimit()) return 12;
        if (!CAL.have) return -1;
        int best = INT_MAX, second = INT_MAX, bi = -1;
        for (int i = 0; i < 12; ++i) {
            int v = classDist(d, i / 6, i % 6);
            if (v < best) { second = best; best = v; bi = i; }
            else if (v < second) second = v;
        }
        miss = best;
        gap = second - best;
        if (best > 2200 || gap < 200) return -1;
        return bi;
    }

    // Returns the number of squares that had to be guessed. The site sometimes
    // leaves a piece drawn half off the board after an animation, and refusing
    // to read the position at all would park the engine until the clock runs
    // out; a guessed square at worst costs one refused move.
    static int readBoard(const Shot& s, array<int, 64>& out) {
        int doubtful = 0;
        for (int cell = 0; cell < 64; ++cell) {
            Desc d = cellDesc(s, cell);
            int miss, gap;
            int v = classify(d, miss, gap);
            if (v >= 0) { out[cell] = v; continue; }
            doubtful++;
            int best = INT_MAX, bi = 12;
            for (int i = 0; i < 12; ++i) {
                int q = classDist(d, i / 6, i % 6);
                if (q < best) { best = q; bi = i; }
            }
            out[cell] = bi;
        }
        return doubtful;
    }

    // The dice always show the drawings of the side to move, so their colour is
    // known from the clocks and never has to be guessed: matching against six
    // shapes instead of twelve roughly triples the margin, which matters here
    // because the outline on a die is thicker than on the board.
    // Mean brightness of a die's red face, ignoring the drawing on it. A die
    // that has been spent, or that has no legal move right now, is drawn dimmed.
    static int dieGlow(const Shot& s, int i) {
        int x0 = dieX(i), y0 = dieY(), n = dieSize(), in = rel(16);
        long long sum = 0;
        int cnt = 0;
        for (int y = in; y < n - in; ++y) for (int x = in; x < n - in; ++x) {
            uint32_t v = s.at(x0 + x, y0 + y);
            int lum;
            if (isInk(v, lum)) continue;
            sum += (int)(v & 255u) + (int)((v >> 8) & 255u) + (int)((v >> 16) & 255u);
            cnt++;
        }
        return cnt ? (int)(sum / (3 * cnt)) : 0;
    }

    static bool readDice(const Shot& s, int colour, array<int, 3>& types, long long* totalMiss = nullptr) {
        if (!CAL.have || colour < 0 || colour > 1) return false;
        long long sum = 0;
        for (int i = 0; i < 3; ++i) {
            Desc d = diceDesc(s, i);
            if (d.ink < 500) return false;
            int best = INT_MAX, second = INT_MAX, bt = -1;
            for (int t = 0; t < 6; ++t) {
                int v = classDist(d, colour, t);
                if (v < best) { second = best; best = v; bt = t; }
                else if (v < second) second = v;
            }
            if (best > 4000 || second - best < 400) return false;
            types[i] = bt;
            sum += best;
        }
        if (totalMiss) *totalMiss = sum;
        return true;
    }

    // Used only when joining a game in progress. The dice cannot say which
    // colour we are - a white and a black drawing of the same piece differ by
    // almost nothing once the shape is normalised - but the board can: our own
    // men sit on our own side of it.
    // The rank label in the bottom-left corner says which way the board is
    // drawn: "1" there means White is at the bottom, "8" means Black is. The
    // label is a shade of the wood rather than ink, so it is found by how much
    // it differs from the square's own colour; "1" measures about 11 px across,
    // "8" is half again as wide. Unlike counting pieces this works in an
    // endgame, where there may be nothing left to count.
    static int colourFromCornerDigit(const Shot& s, int* outWidth = nullptr, int* outN = nullptr) {
        const int x0 = BX + rel(4), y0 = BY + 7 * CELL + rel(8);
        const int w = rel(22), h = rel(30);
        int hist[4096] = { 0 };
        auto bucket = [](uint32_t v) {
            return (int)(((v >> 20) & 15) | (((v >> 12) & 15) << 4) | (((v >> 4) & 15) << 8));
            };
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) hist[bucket(s.at(x0 + x, y0 + y))]++;
        int bg = 0, best = -1;
        for (int i = 0; i < 4096; ++i) if (hist[i] > best) { best = hist[i]; bg = i; }
        // Average the pixels of the winning bucket rather than reconstructing a
        // colour from the bucket index: the index is quantised to 16 levels, and
        // the rounding error alone exceeds the threshold below, which made every
        // pixel look unlike the background.
        long long sR = 0, sG = 0, sB = 0, cnt = 0;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                uint32_t v = s.at(x0 + x, y0 + y);
                if (bucket(v) != bg) continue;
                sB += (int)(v & 255u); sG += (int)((v >> 8) & 255u); sR += (int)((v >> 16) & 255u);
                ++cnt;
            }
        if (!cnt) return -1;
        const int bR = (int)(sR / cnt), bG = (int)(sG / cnt), bB = (int)(sB / cnt);

        int n = 0, ax = 1 << 30, zx = -1;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                uint32_t v = s.at(x0 + x, y0 + y);
                const int B = (int)(v & 255u), G = (int)((v >> 8) & 255u), R = (int)((v >> 16) & 255u);
                if (abs(R - bR) + abs(G - bG) + abs(B - bB) < 40) continue;
                ++n;
                if (x < ax) ax = x;
                if (x > zx) zx = x;
            }
        const int width = (zx >= ax) ? (zx - ax + 1) : 0;
        if (outWidth) *outWidth = width;
        if (outN) *outN = n;
        if (n < 40 || zx < ax) return -1;
        if (width <= rel(13)) return 0;    // "1": White at the bottom
        if (width >= rel(15)) return 1;    // "8": Black at the bottom
        return -1;
    }

    static bool guessOurColour(const array<int, 64>& b, int& colour) {
        // Which colour sits lower on screen, measured as the mean row of its
        // men. Counting pieces in the bottom rows fails in an endgame - four
        // pieces on the board and there is nothing to count - whereas the mean
        // row still separates the sides cleanly.
        long long sum[2] = { 0,0 };
        int cnt[2] = { 0,0 };
        for (int cell = 0; cell < 64; ++cell) {
            const int v = b[cell];
            if (v < 0 || v >= 12) continue;
            sum[v / 6] += cell >> 3;
            cnt[v / 6]++;
        }
        if (!cnt[0] || !cnt[1]) return false;
        const double avgWhite = (double)sum[0] / cnt[0];
        const double avgBlack = (double)sum[1] / cnt[1];
        if (fabs(avgWhite - avgBlack) < 1.0) return false;
        colour = avgWhite > avgBlack ? 0 : 1;
        return true;
    }

    // ------------------------------------------------------------------ state

    static int clockLit(const Shot& s, int x0, int y0, int x1, int y1) {
        int n = 0;
        for (int y = y0; y < y1; ++y) for (int x = x0; x < x1; ++x) {
            uint32_t v = s.at(x, y);
            int B = (int)(v & 255u), G = (int)((v >> 8) & 255u), R = (int)((v >> 16) & 255u);
            if (R > 170 && G > 165 && B > 140) n++;
        }
        return n;
    }
    static int clockTop(const Shot& s) {
        return clockLit(s, BX + rel(946), BY - rel(180), BX + rel(1160), BY - rel(80));
    }
    static int clockBottom(const Shot& s) {
        return clockLit(s, BX + rel(946), BY + rel(1160), BX + rel(1170), BY + rel(1258));
    }
    // The clock of the side to move is printed in bright cream, the waiting
    // side's is dimmed. Average brightness is not enough: the active clock sits
    // on a dark plaque, so count the pixels above the bright threshold instead.
    static int ourTurn(const Shot& s) {
        int a = clockBottom(s), b = clockTop(s);
        if (a > 60 && a > 3 * b) return 1;
        if (b > 60 && b > 3 * a) return 0;
        return -1;
    }

    // While the site is reconnecting it prints a white "CONNECTING" banner over
    // the top of the page and quietly ignores every click. Measured: about 2700
    // light pixels in that strip with the banner up against about 250 without.
    static int reconnecting(const Shot& s) {
        int n = 0;
        int cx = BX + boardW() / 2;
        for (int y = BY - rel(457); y < BY - rel(327); ++y)
            for (int x = cx - rel(260); x < cx + rel(260); ++x) {
                uint32_t v = s.at(x, y);
                if ((v & 255u) > 200 && ((v >> 8) & 255u) > 200 && ((v >> 16) & 255u) > 200) n++;
            }
        return n > 1000;
    }

    // Dumps a region to a 24-bit BMP. Used to look at the end-of-game panel:
    // it is on screen for about a second, which is awkward to catch by hand.
    static void saveBmp(const Shot& s, int x0, int y0, int w, int h, const char* name) {
        if (w <= 0 || h <= 0) return;
        const int rowBytes = (w * 3 + 3) & ~3;
        const int dataSize = rowBytes * h;
        ofstream f(name, ios::binary);
        if (!f) return;
        unsigned char hdr[54] = { 0 };
        hdr[0] = 'B'; hdr[1] = 'M';
        const int fileSize = 54 + dataSize;
        memcpy(hdr + 2, &fileSize, 4);
        const int off = 54; memcpy(hdr + 10, &off, 4);
        const int hsz = 40;  memcpy(hdr + 14, &hsz, 4);
        memcpy(hdr + 18, &w, 4);
        memcpy(hdr + 22, &h, 4);
        const short planes = 1, bpp = 24;
        memcpy(hdr + 26, &planes, 2); memcpy(hdr + 28, &bpp, 2);
        memcpy(hdr + 34, &dataSize, 4);
        f.write((char*)hdr, 54);
        vector<unsigned char> row(rowBytes, 0);
        for (int y = h - 1; y >= 0; --y) {
            for (int x = 0; x < w; ++x) {
                uint32_t v = s.at(x0 + x, y0 + y);
                row[x * 3 + 0] = (unsigned char)(v & 255u);
                row[x * 3 + 1] = (unsigned char)((v >> 8) & 255u);
                row[x * 3 + 2] = (unsigned char)((v >> 16) & 255u);
            }
            f.write((char*)row.data(), rowBytes);
        }
    }

    // The end-of-game panel states the verdict above the board. "ВЫ ПРОИГРАЛИ"
    // is one letter longer than "ВЫ ВЫИГРАЛИ", so the width of the white
    // caption tells the two apart; the pixel count is logged with it so the
    // threshold can be set from measurement.
    // Reads the verdict off the panel's headline. Both headlines are drawn the
    // same way - "ВЫ" in white, the verdict itself in cream (248,215,185) - so
    // what separates them is the length of that second word: measured 279 px
    // for ВЫИГРАЛИ against 309 px for ПРОИГРАЛИ, one letter apart.
    // Returns 1 won, -1 lost, 0 unreadable.
    static int verdictFromHeader(const Shot& s, int* outWidth = nullptr) {
        const int cx = BX + boardW() / 2;
        int x0 = 1 << 30, x1 = -1, n = 0;
        // Just the headline. A band that reaches lower also picks up the line
        // underneath it ("Сбитый король" / "Закончилось время"), and that line
        // is drawn in the same cream and is the wider of the two - which is how
        // a loss came out as 678 px instead of 309.
        for (int y = BY + rel(265); y < BY + rel(315); ++y)
            for (int x = cx - rel(420); x < cx + rel(420); ++x) {
                uint32_t v = s.at(x, y);
                const int B = (int)(v & 255u), G = (int)((v >> 8) & 255u), R = (int)((v >> 16) & 255u);
                if (R > 230 && G > 195 && G < 232 && B > 160 && B < 205) {
                    ++n;
                    if (x < x0) x0 = x;
                    if (x > x1) x1 = x;
                }
            }
        const int w = (x1 >= x0) ? (x1 - x0) : 0;
        if (outWidth) *outWidth = w;
        if (n < 500) return 0;                       // no headline in view
        if (w >= rel(250) && w < rel(294)) return 1;  // ВЫИГРАЛИ
        if (w >= rel(294) && w < rel(340)) return -1; // ПРОИГРАЛИ
        return 0;
    }

    // The end-of-game panel puts a white "Rematch" caption below the board.
    // Is the board actually on screen? The end-of-game panel covers the middle
    // but never the outer ranks, so the wood there is a reliable sign. Without
    // this the engine happily clicks away at whatever page took the tab's
    // place - it once pressed "rematch" five times into YouTube.
    static bool boardPresent(const Shot& s) {
        int wood = 0, total = 0;
        for (int r = 0; r < 8; r += 7)
            for (int y = BY + r * CELL + 4; y < BY + (r + 1) * CELL - 4; y += 4)
                for (int x = BX + 4; x < BX + boardW() - 4; x += 4) {
                    uint32_t v = s.at(x, y);
                    const int B = (int)(v & 255u), G = (int)((v >> 8) & 255u), R = (int)((v >> 16) & 255u);
                    ++total;
                    if (R - B > 45 && R - B < 95 && R > 130 && G > 100) ++wood;
                }
        return total && wood * 100 / total > 25;
    }

    static int rematchReady(const Shot& s) {
        // The panel covers the middle of the board, so the wood there gives out.
        // Checking the caption alone was not enough: light squares under a move
        // highlight pass for white often enough to fire on their own.
        {
            const int cx = BX + boardW() / 2, cy = BY + boardW() / 2;
            int wood = 0, total = 0;
            for (int y = cy - CELL; y < cy + CELL; y += 3)
                for (int x = cx - CELL; x < cx + CELL; x += 3) {
                    uint32_t v = s.at(x, y);
                    const int B = (int)(v & 255u), G = (int)((v >> 8) & 255u), R = (int)((v >> 16) & 255u);
                    ++total;
                    if (R - B > 45 && R - B < 95 && R > 130) ++wood;
                }
            if (total && wood * 100 / total > 25) return 0;   // board still visible
        }
        int n = 0;
        int x0 = rematchX() - rel(160), x1 = rematchX() + rel(160);
        int y0 = rematchY() - rel(40), y1 = rematchY() + rel(40);
        for (int y = y0; y < y1; ++y) for (int x = x0; x < x1; ++x) {
            uint32_t v = s.at(x, y);
            int B = (int)(v & 255u), G = (int)((v >> 8) & 255u), R = (int)((v >> 16) & 255u);
            if (R > 200 && G > 200 && B > 200) n++;
        }
        return n > 600;
    }

    // ------------------------------------------------------------------ mouse

    static void click(int x, int y) {
        SetCursorPos(x, y);
        Sleep(30);
        mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0);
        Sleep(35);
        mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0);
        Sleep(30);
    }
    static void clickCell(int cell) {
        click(cellX(cell & 7) + CELL / 2, cellY(cell >> 3) + CELL / 2);
    }

    // Last resort when the page has stopped taking clicks: the site sometimes
    // drops its connection and sits there showing a live board that no longer
    // responds, and nothing short of a reload brings it back.
    static void reloadPage() {
        click(BX - rel(60), BY + boardW() / 2);      // focus the page
        Sleep(300);
        keybd_event(VK_F5, 0, 0, 0);
        Sleep(60);
        keybd_event(VK_F5, 0, KEYEVENTF_KEYUP, 0);
    }

    // --------------------------------------------------------- board <-> chess

    // flip = 0: we play White, the board is drawn with a1 bottom-left.
    // flip = 1: we play Black, the board is drawn with h8 bottom-left.
    static int cellOfSq(int sq, int flip) {
        int r = sq >> 3, f = sq & 7;
        return flip ? (r * 8 + (7 - f)) : ((7 - r) * 8 + f);
    }
    static int sqOfCell(int cell, int flip) {
        int r = cell >> 3, f = cell & 7;
        return flip ? (r * 8 + (7 - f)) : ((7 - r) * 8 + f);
    }

    static void boardOfPos(const Position& p, int flip, array<int, 64>& out) {
        out.fill(12);
        for (int sq = 0; sq < 64; ++sq) {
            uint64_t b = bit(sq);
            int c = (p.color[0] & b) ? 0 : ((p.color[1] & b) ? 1 : -1);
            if (c < 0) continue;
            for (int t = 0; t < 6; ++t) if (p.piece[t] & b) { out[cellOfSq(sq, flip)] = c * 6 + t; break; }
        }
    }

    // ---------------------------------------------------------------- learning

    // The initial position is recognisable without any learned data: four full
    // rows of ink at the edges and four empty rows in the middle.
    static bool looksInitial(const Shot& s, array<Desc, 64>& d, int& flip) {
        long long lumSum = 0;
        int lumN = 0;
        for (int cell = 0; cell < 64; ++cell) {
            d[cell] = cellDesc(s, cell);
            int r = cell >> 3;
            if (r <= 1 || r >= 6) {
                if (d[cell].ink < 1200) return false;
                if (r >= 6) {
                    int L = centreLum(s, cell);
                    if (L >= 0) { lumSum += L; lumN++; }
                }
            }
            else if (d[cell].ink > 400) return false;
        }
        if (lumN < 8) return false;
        flip = (lumSum / lumN) > 128 ? 0 : 1;   // bright fill at the bottom = we are White
        return true;
    }

    static void learn(const array<Desc, 64>& d, int flip) {
        // Back rank as it appears on screen, left to right.
        static const int normal[8] = { 3,1,2,4,5,2,1,3 };   // a..h: R N B Q K B N R
        static const int mirror[8] = { 3,1,2,5,4,2,1,3 };   // h..a
        const int* order = flip ? mirror : normal;
        const int us = flip ? 1 : 0;
        const int them = 1 - us;

        Cal c;
        // Back-rank pieces are kept as separate prototypes; the eight pawns of
        // a colour are identical drawings and get averaged into one.
        auto push = [&](int colour, int type, const Desc& src) {
            int& n = c.n[colour][type];
            if (n < MAXPROTO) c.proto[colour][type][n++] = src;
            };
        struct Acc { long long g[36] = { 0 }; long long centre = 0, aspect = 0, ink = 0; int n = 0; };
        Acc pawn[2];
        auto addPawn = [&](int colour, const Desc& src) {
            Acc& a = pawn[colour];
            for (int i = 0; i < 36; ++i) a.g[i] += src.g[i];
            a.centre += src.centre; a.aspect += src.aspect; a.ink += src.ink; a.n++;
            };
        for (int f = 0; f < 8; ++f) {
            push(them, order[f], d[0 * 8 + f]);   // opponent back rank (top)
            push(us, order[f], d[7 * 8 + f]);   // our back rank (bottom)
            addPawn(them, d[1 * 8 + f]);
            addPawn(us, d[6 * 8 + f]);
        }
        for (int col = 0; col < 2; ++col) {
            Acc& a = pawn[col];
            if (!a.n) return;
            Desc p;
            for (int i = 0; i < 36; ++i) p.g[i] = (int)(a.g[i] / a.n);
            p.centre = (int)(a.centre / a.n);
            p.aspect = (int)(a.aspect / a.n);
            p.ink = (int)(a.ink / a.n);
            c.n[col][0] = 0;
            push(col, 0, p);
        }
        for (int col = 0; col < 2; ++col) for (int t = 0; t < 6; ++t) if (!c.n[col][t]) return;

        c.emptyInk = 0;
        for (int cell = 16; cell < 48; ++cell) c.emptyInk = max(c.emptyInk, d[cell].ink);
        c.have = 1;
        CAL = c;
    }

    // ------------------------------------------------------------ diagnostics

    static void dumpBoard(const array<int, 64>& b) {
        static const char* names = "PNBRQKpnbrqk.";
        for (int r = 0; r < 8; ++r) {
            string line;
            for (int f = 0; f < 8; ++f) {
                int v = b[r * 8 + f];
                line += (v < 0 || v > 12) ? '?' : names[v];
                line += ' ';
            }
            cout << "   " << line << '\n';
        }
    }

    static void diagnose(int seconds) {
        loadGeometry();
        loadCal();
        cout << "board " << BX << ',' << BY << " cell " << CELL
            << " | dice y " << dieY() << " x " << dieX(0) << '/' << dieX(1) << '/' << dieX(2)
            << " size " << dieSize()
            << " | promo y " << promoY() << " x " << promoX(0) << ".." << promoX(3)
            << " | rematch " << rematchX() << ',' << rematchY() << '\n';
        auto t0 = steady_clock::now();
        while (duration_cast<chrono::seconds>(steady_clock::now() - t0).count() < seconds) {
            Shot s = grabAll();
            array<Desc, 64> d;
            int flip = -1;
            bool init = looksInitial(s, d, flip);
            cout << "clock top=" << clockTop(s) << " bottom=" << clockBottom(s)
                << " turn=" << ourTurn(s)
                << " rematch=" << rematchReady(s)
                << " initial=" << init << " flip=" << flip
                << " cal=" << CAL.have << " emptyInk=" << CAL.emptyInk << '\n';
            if (init) { learn(d, flip); cout << "learned from the initial position\n"; }
            array<int, 64> b;
            cout << "board doubtful=" << readBoard(s, b) << '\n';
            dumpBoard(b);
            for (int r = 0; r < 8; ++r) {
                cout << "   ";
                for (int f = 0; f < 8; ++f) {
                    Desc dd = cellDesc(s, r * 8 + f);
                    int miss, gap;
                    classify(dd, miss, gap);
                    cout << setw(5) << dd.ink << ' ' << setw(5) << miss << ':' << setw(5) << gap << ' ';
                }
                cout << '\n';
            }
            int dc = -1;
            int cdW = 0, cdN = 0;
            const int cd = colourFromCornerDigit(s, &cdW, &cdN);
            cout << "corner digit says=" << cd << " (w=" << cdW << " n=" << cdN << ")"
                << " piece spread says=" << (guessOurColour(b, dc) ? dc : -1);
            for (int c = 0; c < 2; ++c) {
                array<int, 3> t{}; long long mm = 0;
                bool o = readDice(s, c, t, &mm);
                cout << " | col" << c << " ok=" << o << " miss=" << mm;
                if (o) { cout << " ["; for (int i = 0; i < 3; ++i) cout << pieceChar(t[i]); cout << ']'; }
            }
            cout << " | glow " << dieGlow(s, 0) << '/' << dieGlow(s, 1) << '/' << dieGlow(s, 2);
            for (int i = 0; i < 3; ++i) {
                Desc dd = diceDesc(s, i);
                cout << "\n   die#" << i << " ink=" << dd.ink << " centre=" << dd.centre << " aspect=" << dd.aspect << ":";
                for (int c = 0; c < 2; ++c) for (int t = 0; t < 6; ++t)
                    cout << ' ' << (c ? (char)toupper(pieceChar(t)) : pieceChar(t)) << classDist(dd, c, t);
            }
            for (int i = 0; i < 3; ++i) {
                Desc dd = diceDesc(s, i);
                int miss, gap;
                int v = classify(dd, miss, gap);
                cout << "  #" << i << " ink=" << dd.ink << " -> " << v
                    << " miss=" << miss << " gap=" << gap;
            }
            cout << "\n---\n";
            Sleep(1500);
        }
    }

    // ------------------------------------------------------------------- play

    struct Player {
        array<uint64_t, 4> path;
        array<int, 64> mask;
        int flip = -1;
        int us = 0;                  // 0 = white
        int castle = 15;
        array<int, 64> lastBoard;    // board as we left it at the end of our turn
        int haveLast = 0;
        // Record of the game in progress: one entry per turn holding how many
        // moves that roll bought, plus how the game ended.
        vector<int> series;
        int outcome = 0;             // 1 won, -1 lost, 0 undecided (clock, most likely)
        time_t started = 0;
        // En passant rights seen appearing during the opponent's turn, and the
        // last board read while watching it.
        uint64_t epPending = 0;
        array<int, 64> watchBoard{};
        int haveWatch = 0;
    };

    // Opened fresh at the start of every run, so the file always holds the
    // results of the session in front of you and nothing older.
    static ofstream g_results;

    // One character per finished game, in order: W won, L lost. Nothing else -
    // the sequence is all that is needed to work out a win rate.
    static void logGame(const Player& pl) {
        if (pl.outcome == 0) return;
        // A panel that was already there when we started belongs to a game we
        // did not play, and its verdict is not ours to record. One that appears
        // after we have seen a live board is.
        if (g_results) g_results << (pl.outcome > 0 ? 'W' : 'L') << flush;

        static long long total = 0, wins = 0, streak = 0;
        ++total;
        if (pl.outcome > 0) { ++wins; ++streak; }
        else streak = 0;
        if (total % 100 == 0)
            cout << "[stats] " << total << " games, win rate "
            << fixed << setprecision(1) << (100.0 * wins / total) << "%\n"
            << setprecision(6);
        if (streak >= 15)
            cout << "[stats] winning streak of " << streak << '\n';
    }

    // Read the board until two consecutive readings agree. A couple of guessed
    // squares are tolerated once the clean readings have been given a chance.
    static bool stableBoard(array<int, 64>& out, Shot& outShot, int tries) {
        array<int, 64> prev;
        bool havePrev = false;
        for (int i = 0; i < tries; ++i) {
            Shot s = grabAll();
            array<int, 64> b;
            int doubtful = readBoard(s, b);
            int allow = (i < tries * 2 / 3) ? 0 : 2;
            if (doubtful <= allow) {
                if (havePrev && b == prev) { out = b; outShot = s; return true; }
                prev = b; havePrev = true;
            }
            else havePrev = false;
            Sleep(120);
        }
        return false;
    }

    // Rights are only ever removed: a king or rook that left its home square
    // never restores the right by coming back.
    static int castleFromBoard(const array<int, 64>& b, int flip, int have) {
        for (int c = 0; c < 2; ++c) {
            if (b[cellOfSq(4 + 56 * c, flip)] != c * 6 + 5) have &= ~(3 << (2 * c));
            for (int r = 0; r < 2; ++r)
                if (b[cellOfSq(7 * r + 56 * c, flip)] != c * 6 + 3) have &= ~(1 << (r + 2 * c));
        }
        return have;
    }

    static void posFromBoard(Position& p, const array<int, 64>& b, int flip, int side,
        int castle, int diceVal, uint64_t epForSide) {
        p.color = { 0,0 };
        p.piece = { 0,0,0,0,0,0 };
        for (int cell = 0; cell < 64; ++cell) {
            int v = b[cell];
            if (v < 0 || v >= 12) continue;
            int sq = sqOfCell(cell, flip);
            p.color[v / 6] |= bit(sq);
            p.piece[v % 6] |= bit(sq);
        }
        p.side = side;
        p.ep1 = { 0,0 };
        p.ep1[side] = epForSide;
        p.ep2 = 0;
        p.rook = { 0,7,56,63 };
        p.castle = castle;
        p.dice = diceVal;
        p.key = computeKey(p);
    }

    // The site shows the three rolled faces; the engine drops faces for piece
    // types the side to move does not own (see makeRandomWithRolledDice).
    static int normalizeDice(const Position& p, int diceVal) {
        uint64_t pawns = p.color[p.side] & p.piece[0];
        int dist = 6;
        if (pawns) dist = (p.side == 0) ? (clz64(pawns) >> 3) : (ctz64(pawns) >> 3);
        for (int i = 0; i < 5; ++i)
            while (dicePiece[diceVal][i] && (p.color[p.side] & p.piece[i]) == 0 && dist > dicePiece[diceVal][0])
                diceVal = newDice[diceVal][i];
        return diceVal;
    }

    // What changed between two frames, in board coordinates. This is the only
    // window onto the opponent's turn, so it is worth being able to read it.
    static string stepText(const array<int, 64>& prev, const array<int, 64>& now, int flip) {
        static const char* const kPiece = "pnbrqkPNBRQK";
        string out;
        for (int sq = 0; sq < 64; ++sq) {
            const int cell = cellOfSq(sq, flip);
            if (prev[cell] == now[cell]) continue;
            out += ' ';
            if (prev[cell] >= 0 && prev[cell] < 12) out += kPiece[prev[cell]];
            out += sqName(sq);
            out += '>';
            if (now[cell] >= 0 && now[cell] < 12) out += kPiece[now[cell]]; else out += '.';
        }
        return out;
    }

    static uint64_t epFromStep(const array<int, 64>& prev, const array<int, 64>& now,
        int flip, int us) {
        const int them = 1 - us;
        const int pawn = them * 6 + 0;
        const int homeRank = (them == 0) ? 1 : 6;
        const int landRank = (them == 0) ? 3 : 4;
        uint64_t out = 0;
        for (int file = 0; file < 8; ++file) {
            const int fromSq = homeRank * 8 + file;
            const int toSq = landRank * 8 + file;
            const int midSq = (fromSq + toSq) / 2;
            const int fromCell = cellOfSq(fromSq, flip);
            const int toCell = cellOfSq(toSq, flip);
            const int midCell = cellOfSq(midSq, flip);
            if (prev[fromCell] != pawn || now[fromCell] != 12) continue;
            if (prev[toCell] != 12 || now[toCell] != pawn) continue;
            if (prev[midCell] != 12 || now[midCell] != 12) continue;   // nothing paused there
            out |= bit(midSq);
        }
        return out;
    }

    static uint64_t epFromDiff(const array<int, 64>& before, const array<int, 64>& after,
        int flip, int us) {
        // Examined file by file rather than by counting changed squares. A turn
        // here can hold several moves, so demanding that exactly two squares
        // differ threw the right away whenever the opponent did anything else
        // alongside the double push - and the rights accumulate, so three pawns
        // pushing two squares really does create three of them.
        const int them = 1 - us;
        const int pawn = them * 6 + 0;
        const int homeRank = (them == 0) ? 1 : 6;
        const int landRank = (them == 0) ? 3 : 4;
        uint64_t out = 0;
        for (int file = 0; file < 8; ++file) {
            const int fromSq = homeRank * 8 + file;
            const int toSq = landRank * 8 + file;
            const int fromCell = cellOfSq(fromSq, flip);
            const int toCell = cellOfSq(toSq, flip);
            if (before[fromCell] != pawn || after[fromCell] != 12) continue;
            if (before[toCell] != 12 || after[toCell] != pawn) continue;
            out |= bit((fromSq + toSq) / 2);
        }
        return out;
    }

    // How much of a square's picture changed between two captures. The site
    // does not mark the squares a piece may go to, but it does tint the square
    // of a piece it accepts as selected - and it ignores a click on a piece
    // that has no move, which is exactly the case the engine gets wrong.
    static int cellChange(const Shot& a, const Shot& b, int cell) {
        int x0 = cellX(cell & 7), y0 = cellY(cell >> 3), n = 0;
        for (int y = rel(8); y < CELL - rel(8); y += 2) for (int x = rel(8); x < CELL - rel(8); x += 2) {
            uint32_t p = a.at(x0 + x, y0 + y), q = b.at(x0 + x, y0 + y);
            int d = abs((int)(p & 255u) - (int)(q & 255u))
                + abs((int)((p >> 8) & 255u) - (int)((q >> 8) & 255u))
                + abs((int)((p >> 16) & 255u) - (int)((q >> 16) & 255u));
            if (d > 40) n++;
        }
        return n;
    }

    static int g_hintBlind = 0;   // set once the site turns out not to draw hints
    static int g_hintMiss = 0;

    // Play one engine move with the mouse and wait for the site to accept it.
    // Returns 1 = accepted, 0 = the board went somewhere we did not expect
    // (rebuild from the screen), -1 = the site kept refusing the move,
    // 2 = the site will not pick that piece up, pick another move,
    // 3 = the connection dropped; wait for it and repeat the same move.
    static int playMove(const Position& before, const Position& after, int move, int flip) {
        int from = move & 63, to = (move >> 6) & 63, promo = (move >> 12) & 7;
        int target = to;
        if (before.color[before.side] & bit(to)) {
            // Castling is encoded as "king captures its own rook"; the site
            // wants the king dropped on c1/g1/c8/g8.
            int rank = (to >= 56) ? 56 : 0;
            target = rank + (to > from ? 6 : 2);
        }
        array<int, 64> want, had;
        boardOfPos(after, flip, want);
        boardOfPos(before, flip, had);

        // Squares whose occupancy the move changes - the only thing worth
        // watching. The site tints the squares of the move it just played, and
        // a piece on a tinted square can fall short of the recognition margin,
        // but "the square the piece left is now empty" survives any tint.
        vector<int> watch;
        for (int cell = 0; cell < 64; ++cell)
            if ((want[cell] == 12) != (had[cell] == 12)) watch.push_back(cell);

        for (int attempt = 0; attempt < 3; ++attempt) {
            if (attempt) {
                // Drop any half-made selection before trying again.
                click(BX - rel(60), BY + boardW() / 2);
                Sleep(500);
            }
            const int fromCell = cellOfSq(from, flip), toCell = cellOfSq(target, flip);
            // No pre-emptive click beside the board: after a completed move the
            // site clears the selection by itself, and parking the pointer there
            // every time only makes the cursor jump about. A stale selection is
            // dealt with on the retry path below.
            Shot idle = grabAll();

            // Pick the piece up. A click can simply be dropped while the site is
            // still settling, so retry before concluding the piece cannot move.
            bool selected = false;
            for (int k = 0; k < 3 && !selected; ++k) {
                clickCell(fromCell);
                Sleep(260);
                Shot sel = grabAll();
                selected = cellChange(idle, sel, fromCell) >= 300;
            }
            if (!selected) {
                // A dropped connection looks exactly like a piece that cannot
                // move, so ask the page which one it is before giving up on the
                // move: the move itself is fine, the site just is not listening.
                if (reconnecting(grabAll())) return 3;
                if (++g_hintMiss >= 25) { g_hintBlind = 1; cout << "[play] selection unreadable, trusting the engine\n"; }
                if (!g_hintBlind) return 2;
            }
            g_hintMiss = 0;

            Sleep(220);
            clickCell(toCell);
            if (promo) { Sleep(600); click(promoX(promo - 1), promoY()); }

            auto t0 = steady_clock::now();
            int nudges = 0;
            for (;;) {
                Sleep(120);
                Shot s = grabAll();
                int hits = 0, misses = 0;
                for (int cell : watch) {
                    bool isEmpty = cellDesc(s, cell).ink <= inkEmptyLimit();
                    if (isEmpty == (want[cell] == 12)) hits++;
                    else if (isEmpty != (had[cell] == 12)) misses++;
                }
                // The site drops further clicks until the server has confirmed
                // this move, so the wait has to outlast the round trip. It runs
                // in the mouse thread while the next position is searched, so
                // it costs no thinking time.
                if (hits == (int)watch.size()) { Sleep(1100); return 1; }
                // The board moved, but not where we aimed: clicking again would
                // only make it worse, so rebuild the position from the screen.
                if (misses) {
                    cout << "[play] " << moveToStr(move) << ": board changed unexpectedly\n";
                    return 0;
                }
                if (!g_hintBlind && hits == 0) {
                    const bool stillHeld = cellChange(idle, s, fromCell) >= 300;
                    // The site drops the selection when it turns a destination
                    // down, and the square goes back to how it looked. That is a
                    // definite "no" and needs no waiting.
                    if (!stillHeld) return 2;
                    // Still holding the piece after a second means the click on
                    // the destination went missing; another one is cheaper than
                    // starting the whole move over.
                    if (nudges < 2 &&
                        duration_cast<milliseconds>(steady_clock::now() - t0).count() > 900 + 1200 * nudges) {
                        clickCell(toCell);
                        if (promo) { Sleep(600); click(promoX(promo - 1), promoY()); }
                        nudges++;
                    }
                }
                // Waiting on the server can take a while; only give up once the
                // position has been standing still, unchanged, for long enough.
                if (duration_cast<milliseconds>(steady_clock::now() - t0).count() > 5000) break;
            }
            cout << "[play] " << moveToStr(move) << " was not accepted, retrying\n";
        }
        return -1;
    }

    static void run(double firstSec) {
        // Progress goes to the console; the only file this mode writes is
        // results.txt, one character per finished game, started anew each run.
        g_results.open("results.txt", ios::trunc);
        loadGeometry();
        loadCal();
        START();
        Player pl;
        pl.path = PATH;
        pl.mask = MASK;
        pl.lastBoard.fill(12);
        size_t nodeCap, edgeCap;
        tableSizeForTime(firstSec, nodeCap, edgeCap);
        cout << "[play] tree " << nodeCap << " nodes / " << edgeCap << " edges\n";
        MCTSTable T(nodeCap, edgeCap);

        // One inference server for the whole session. Starting and draining it
        // per move costs real thinking time - the batches have to ramp up again
        // every time - and a one-second search is short enough for that to hurt.
        InferenceServer nn(T, &g_trt, g_trt2Ready ? &g_trt2 : nullptr);
        nn.start();
        struct Stopper {
            InferenceServer& s;
            ~Stopper() noexcept { try { s.stopAndDrain(); } catch (...) {} }
        } stopper{ nn };

        cout << "[play] board " << BX << ',' << BY << " cell " << CELL
            << ", analysis " << firstSec << " s after the roll and 1 s inside a turn\n";

        int idle = 0, rejects = 0, banned = 0, sawTheirTurn = 0, noMove = 0;
        // A board seen but not yet believed, and since when. Only a reading that
        // stays put becomes the next watched position.
        array<int, 64> pendBoard{};
        int havePend = 0, pendSeen = 0;
        auto pendSince = steady_clock::now();
        auto pendLast = pendSince;
        // Set once a live board has been seen, so a panel that was on screen
        // from the outset is recognised as belonging to an earlier game.
        bool sawLiveBoard = false;
        auto lastMoveAt = steady_clock::now();   // for timing the gap between moves
        // Survives a broken-off series: rebuilding the position from the screen
        // still leaves us inside the same turn, which has already had its think.
        bool longUsed = false;
        auto lastAccepted = steady_clock::now();
        for (;;) try {
            Shot s = grabAll();

            // Nothing at all happens unless the board is in front of us.
            if (!boardPresent(s)) {
                if (++idle % 60 == 0) cout << "[wait] board is not on screen\n";
                pl.flip = -1;
                pl.haveLast = 0;
                lastAccepted = steady_clock::now();
                Sleep(500);
                continue;
            }

            if (reconnecting(s)) {
                if (++idle % 20 == 0) cout << "[play] connection lost, waiting\n";
                lastAccepted = steady_clock::now();
                Sleep(1000);
                continue;
            }

            if (!rematchReady(s)) sawLiveBoard = true;   // a game is actually on

            if (rematchReady(s)) {
                int hw = 0;
                int said = verdictFromHeader(s, &hw);
                // Give the panel time to be readable before dismissing it -
                // clicking rematch first throws the result away.
                for (int wait = 0; !said && wait < 20; ++wait) {
                    Sleep(250);
                    Shot again = grabAll();
                    if (!rematchReady(again)) break;      // panel gone on its own
                    said = verdictFromHeader(again, &hw);
                }
                if (said) pl.outcome = said;   // the panel is the authority
                cout << "[play] game over ("
                    << (pl.outcome > 0 ? "won" : pl.outcome < 0 ? "lost" : "no verdict")
                    << ", headline " << hw << "px)"
                    << (sawLiveBoard ? "" : ", not our game") << ", clicking rematch\n";
                if (sawLiveBoard) logGame(pl);
                sawLiveBoard = false;
                pl.series.clear();
                pl.outcome = 0;
                click(rematchX(), rematchY());
                pl.flip = -1; pl.castle = 15; pl.haveLast = 0;
                pl.epPending = 0; pl.haveWatch = 0;
                T.newGame();
                // Catch the board the instant the panel clears. The opponent
                // can push a pawn two squares within a few hundred
                // milliseconds, and a double push not seen while it happens is
                // a right lost for good - there is nothing in the position
                // afterwards that distinguishes it from two single steps. So
                // no sleeping here: poll until the board reads, take that
                // frame as the baseline, and take the colour from the corner
                // digit at the same time, since watching needs to know which
                // side we are.
                // Two frames matter here and they are not the same one. The
                // earliest readable frame is the closest we ever get to the
                // position before the opponent touched it. The first frame that
                // holds still is the first we can trust - taken alone, the
                // earliest one can be a piece caught mid-slide and would put a
                // pawn on a square it never stopped on. Keep both: the step
                // between them is a sighting like any other, and for a game that
                // opens with a double push it is the only one there will be.
                array<int, 64> firstSeen{}, settling{};
                int haveFirst = 0, haveSettling = 0, firstWait = 0;
                auto settleSince = steady_clock::now();
                for (int wait = 0; wait < 200; ++wait) {
                    Shot n = grabAll();
                    array<int, 64> b;
                    if (!boardPresent(n) || rematchReady(n) || readBoard(n, b) != 0) {
                        Sleep(20);
                        continue;
                    }
                    const int dc = colourFromCornerDigit(n);
                    if (dc < 0) { Sleep(20); continue; }
                    const auto now = steady_clock::now();
                    if (!haveFirst) { firstSeen = b; haveFirst = 1; firstWait = wait; }
                    if (!haveSettling || b != settling) {
                        settling = b; haveSettling = 1; settleSince = now;
                        Sleep(20);
                        continue;
                    }
                    if (duration_cast<chrono::milliseconds>(now - settleSince).count() < 200) {
                        Sleep(20);
                        continue;
                    }
                    pl.flip = dc; pl.us = dc;
                    pl.started = time(nullptr);
                    if (b != firstSeen) {
                        const uint64_t got = epFromStep(firstSeen, b, pl.flip, pl.us);
                        pl.epPending |= got;
                        cout << "[watch]" << stepText(firstSeen, b, pl.flip) << " (opening)";
                        if (got) cout << " -> ep " << sqName(ctz64(got));
                        cout << '\n';
                    }
                    pl.watchBoard = b; pl.haveWatch = 1;
                    cout << "[play] new game, we play " << (pl.us ? "black" : "white")
                        << ", first frame after " << firstWait << " polls, settled after "
                        << wait << ", turn=" << ourTurn(n) << '\n';
                    break;
                }
                idle = 0; rejects = 0; havePend = 0;
                continue;
            }

            if (pl.flip < 0) {
                array<Desc, 64> d;
                int flip = -1;
                if (looksInitial(s, d, flip)) {
                    Sleep(250);
                    Shot s2 = grabAll();
                    array<Desc, 64> d2;
                    int flip2 = -1;
                    if (looksInitial(s2, d2, flip2) && flip2 == flip) {
                        learn(d2, flip);
                        pl.flip = flip;
                        pl.us = flip;
                        pl.castle = 15;
                        pl.haveLast = 0;
                        pl.series.clear();
                        pl.outcome = 0;
                        pl.started = time(nullptr);
                        cout << "[play] new game, we play " << (pl.us ? "black" : "white") << '\n';
                    }
                }
                else {
                    // Joining a game already in progress: the dice always show
                    // the drawings of the side to move, so when our clock is
                    // running their colour is our colour.
                    // The corner label is the primary answer; the spread of the
                    // pieces is only a fallback if the label cannot be read.
                    array<int, 64> b;
                    int dc = colourFromCornerDigit(s);
                    if (dc < 0) { readBoard(s, b); if (!guessOurColour(b, dc)) dc = -1; }
                    // Do not wait for our clock to start. Knowing the colour is
                    // what switches on watching the opponent's turn, and a
                    // double push made before we look is a right lost for good.
                    if (CAL.have && dc >= 0) {
                        pl.flip = dc;
                        pl.us = dc;
                        pl.castle = 15;
                        pl.haveLast = 0;
                        pl.epPending = 0;
                        // Start the baseline right away rather than on the next
                        // pass: whatever the opponent does from here on is
                        // watchable, and a frame skipped is a right lost.
                        pl.haveWatch = (readBoard(s, b) == 0);
                        if (pl.haveWatch) pl.watchBoard = b;
                        cout << "[play] joined a running game, we play "
                            << (pl.us ? "black" : "white") << '\n';
                        continue;
                    }
                    if (++idle % 40 == 0) cout << "[play] waiting for the initial position\n";
                }
                Sleep(250);
                continue;
            }

            // Our turn for minutes on end with nothing going through means the
            // page has stopped listening; only a reload gets the game back.
            if (duration_cast<chrono::seconds>(steady_clock::now() - lastAccepted).count() > 180) {
                cout << "[play] no move has gone through for three minutes, reloading the page\n";
                reloadPage();
                pl.flip = -1; pl.haveLast = 0;
                T.newGame();
                lastAccepted = steady_clock::now();
                Sleep(15000);
                continue;
            }

            if (ourTurn(s) != 1) {
                if (ourTurn(s) == 0) { sawTheirTurn = 1; lastAccepted = steady_clock::now(); }
                // Watch for a king going off the board while the opponent moves.
                // The end-of-game panel appears immediately afterwards and hides
                // the middle of the board, so by the time it is up there is no
                // reading the verdict off the position any more.
                if (pl.flip >= 0 && CAL.have) {
                    ++idle;
                    array<int, 64> b;
                    if (readBoard(s, b) == 0) {
                        // Follow the turn as it is played: a double push is only
                        // recognisable while it happens, not from the position
                        // the turn ends in.
                        //
                        // But a frame is not a position until it has held still.
                        // The site slides the piece across the board, so a frame
                        // caught halfway through d7-d5 shows the pawn sitting on
                        // d6 - indistinguishable, taken at face value, from a
                        // turn that really did play d7-d6 and then d6-d5. Believe
                        // that frame and the double push is read as two single
                        // steps and the right is thrown away for nothing. A real
                        // stop on the square outlasts any slide over it.
                        if (b != pl.watchBoard) {
                            const auto now = steady_clock::now();
                            // A turn moves a piece or two. When most of the board
                            // is different it is not a move at all - the site has
                            // dealt a new game while we were busy elsewhere, and
                            // reading a move out of that gives a king crossing
                            // the board in one step. Start over instead.
                            int changed = 0;
                            for (int cell = 0; cell < 64; ++cell)
                                if (b[cell] != pl.watchBoard[cell]) ++changed;
                            if (pl.haveWatch && changed > 4) {
                                cout << "[play] the board was dealt anew (" << changed
                                    << " squares differ), starting the watch over\n";
                                pl.watchBoard = b;
                                pl.epPending = 0;
                                pl.haveLast = 0;
                                pl.castle = 15;
                                havePend = 0;
                                Sleep(50);
                                continue;
                            }
                            if (!havePend || b != pendBoard) {
                                // Report the span between the first and the LAST
                                // reading that showed this frame, never the time
                                // up to the reading that replaced it: the latter
                                // includes a whole polling interval nobody
                                // looked at, and it makes a slide caught twice
                                // look as long-lived as a real stop. The count of
                                // readings goes with it, since a span measured
                                // over two samples is only as sharp as the gap
                                // between them.
                                if (havePend)
                                    cout << "[flick]" << stepText(pl.watchBoard, pendBoard, pl.flip)
                                    << " held "
                                    << duration_cast<chrono::milliseconds>(pendLast - pendSince).count()
                                    << " ms over " << pendSeen << " reads\n";
                                pendBoard = b;
                                havePend = 1;
                                pendSince = now;
                                pendLast = now;
                                pendSeen = 1;
                            }
                            else {
                                pendLast = now;
                                ++pendSeen;
                                if (duration_cast<chrono::milliseconds>(pendLast - pendSince).count() >= 200) {
                                    if (pl.haveWatch) {
                                        const uint64_t got = epFromStep(pl.watchBoard, b, pl.flip, pl.us);
                                        pl.epPending |= got;
                                        cout << "[watch]" << stepText(pl.watchBoard, b, pl.flip)
                                            << " held "
                                            << duration_cast<chrono::milliseconds>(pendLast - pendSince).count()
                                            << " ms over " << pendSeen << " reads";
                                        if (got) cout << " -> ep " << sqName(ctz64(got));
                                        cout << '\n';
                                    }
                                    pl.watchBoard = b;
                                    pl.haveWatch = 1;
                                    havePend = 0;
                                }
                            }
                        }
                        else havePend = 0;

                        int kings[2] = { 0, 0 };
                        for (int cell = 0; cell < 64; ++cell)
                            if (b[cell] == 5) kings[0]++; else if (b[cell] == 11) kings[1]++;
                        if (!kings[pl.us]) pl.outcome = -1;
                        else if (!kings[1 - pl.us]) pl.outcome = 1;
                    }
                }
                if (idle % 150 == 0)
                    cout << "[wait] turn=" << ourTurn(s) << " clocks " << clockBottom(s)
                    << '/' << clockTop(s) << '\n';
                // Sample often enough to see a turn move by move. Two single
                // pawn steps look exactly like one double push if the frame
                // between them is missed, and they carry no en passant right.
                // The reading itself costs tens of milliseconds, so barely sleep
                // at all: the finer the sampling, the tighter the bound on how
                // long a frame really held, and that bound is the whole basis for
                // telling a piece in flight from a piece at rest.
                Sleep(10);
                continue;
            }

            // Our turn: let the roll animation settle, then read everything.
            array<int, 64> board;
            Shot ss;
            if (!stableBoard(board, ss, 15)) {
                if (++idle % 8 == 0) {
                    Shot t = grabAll();
                    array<int, 64> b;
                    readBoard(t, b);
                    cout << "[wait] board unreadable\n";
                    dumpBoard(b);
                }
                Sleep(200);
                continue;
            }
            if (ourTurn(ss) != 1) continue;

            array<int, 3> faces{}, faces2{};
            if (!readDice(ss, pl.us, faces)) {
                if (++idle % 8 == 0) {
                    cout << "[wait] dice unreadable:";
                    for (int i = 0; i < 3; ++i) {
                        Desc dd = diceDesc(ss, i);
                        int miss, gap;
                        int v = classify(dd, miss, gap);
                        cout << " #" << i << " ink=" << dd.ink << " ->" << v
                            << " miss=" << miss << " gap=" << gap;
                    }
                    cout << '\n';
                }
                Sleep(200);
                continue;
            }
            Sleep(250);
            Shot ss2 = grabAll();
            if (!readDice(ss2, pl.us, faces2) || faces != faces2) { Sleep(150); continue; }
            idle = 0;

            // A dimmed die is one the site will not let us use - it has either
            // been spent already or has no legal move at the moment. Taking the
            // whole set on faith is what makes a rebuilt position offer moves
            // the site then refuses over and over.
            // A lit face measures around 60, a dimmed one around 28, so the
            // absolute floor matters: with every face dimmed there is no move
            // to make at all and the relative test would call them all live.
            int glow[3], best = 0, live = 0;
            for (int i = 0; i < 3; ++i) { glow[i] = dieGlow(ss2, i); best = max(best, glow[i]); }
            const int lit = max(45, best * 65 / 100);
            for (int i = 0; i < 3; ++i) if (glow[i] >= lit) live++;
            if (!live) { Sleep(400); continue; }

            // At the start of our turn the whole roll is ours even if some
            // faces are dimmed for want of a legal move, and the engine has to
            // see all three to apply the "use as many dice as possible" rule
            // the way the site does. Only when we did not witness the turn
            // change - after a rebuild, mid-series - are dimmed faces spent
            // ones that must be dropped.
            const bool wholeRoll = sawTheirTurn != 0;
            sawTheirTurn = 0;

            pl.castle = castleFromBoard(board, pl.flip, pl.castle);
            // What was actually seen happening outranks what the endpoints
            // suggest; the endpoint comparison only fills in when the turn was
            // not watched, and it cannot tell one double push from two singles.
            uint64_t ep = pl.epPending;
            // The turn can also end between two frames - the last frame watched
            // and the board we are looking at now. That step is as much a
            // sighting as any other, so read it the same way.
            if (pl.haveWatch && board != pl.watchBoard) {
                const uint64_t got = epFromStep(pl.watchBoard, board, pl.flip, pl.us);
                ep |= got;
                cout << "[watch]" << stepText(pl.watchBoard, board, pl.flip);
                if (got) cout << " -> ep " << sqName(ctz64(got));
                cout << " (turn ended)\n";
            }
            if (!ep && pl.haveLast) ep = epFromDiff(pl.lastBoard, board, pl.flip, pl.us);
            pl.epPending = 0;
            pl.haveWatch = 0;
            // Our own turn takes a second of thinking, during which nothing is
            // watched. A frame left half-believed from before that gap is not a
            // frame that held still - it is one nobody looked at.
            havePend = 0;

            string tok, tokLive;
            for (int i = 0; i < 3; ++i) {
                tok += pieceChar(faces[i]);
                if (glow[i] >= lit) tokLive += pieceChar(faces[i]);
            }
            if (!wholeRoll) swap(tok, tokLive);   // report the set we are sure of

            Position pos;
            posFromBoard(pos, board, pl.flip, pl.us, pl.castle, diceFenToInt(tok), ep);
            if ((pos.color[0] & pos.piece[5]) == 0 || (pos.color[1] & pos.piece[5]) == 0) {
                // A missing king is also how a game ends, so note who lost one
                // before backing off - searching such a position crashes the
                // generator.
                if ((pos.color[pl.us] & pos.piece[5]) == 0) pl.outcome = -1;
                else pl.outcome = 1;
                Sleep(500);
                continue;
            }
            pos.dice = normalizeDice(pos, pos.dice);
            pos.key = computeKey(pos);

            // No guessing here: en passant rights come from what was actually
            // seen on the board and nothing else. If the position offers no
            // move, read again rather than invent a right - and keep the
            // "we saw the turn change" flag, since nothing was played.
            {
                MoveList probeMl; int probeTerm = 0;
                Position probe = pos;
                genLegal(probe, pl.path, pl.mask, probeMl, probeTerm);
                if (probeMl.n == 0) {
                    if (++noMove % 20 == 1)
                        cout << "[play] no legal move in the position as read (roll ["
                        << tok << "], ep " << (ep ? "yes" : "no") << ")\n";
                    sawTheirTurn = wholeRoll;
                    Sleep(400);
                    continue;
                }
                noMove = 0;
            }

            // Mid-series we cannot tell a spent die from one that is merely
            // blocked, and the difference matters: the site applies "use as
            // many dice as possible" to everything still in the roll. So carry
            // a second reading with only the lit faces and play moves that are
            // legal under both - those are accepted either way.
            const bool partial = !wholeRoll;
            Position posLive = pos;
            if (partial) {
                posLive.dice = normalizeDice(posLive, diceFenToInt(tokLive));
                posLive.key = computeKey(posLive);
            }

            cout << "[play] roll [" << tok << "] castle=" << pl.castle
                << (ep ? " ep" : "") << (partial ? " partial +[" + tokLive + "]" : "")
                << (ep ? " (watched)" : "")
                << " glow " << glow[0] << '/' << glow[1] << '/' << glow[2] << '\n';

            // The tree serves one roll. It carries over between the positions
            // of a turn, where it is genuinely useful, and is dropped when the
            // dice are thrown again - everything in it then describes a roll
            // that will not happen.
            // The long think belongs to the roll, not to the attempt: a series
            // rebuilt from the screen mid-turn has already had it.
            if (wholeRoll) longUsed = false;

            if (wholeRoll) T.newGame();
            else {
                const bool aborted = T.abort.load(memory_order_relaxed);
                const double fill = (double)T.edgeTop.load(memory_order_relaxed) / (double)T.edges.size();
                if (aborted || fill > 0.75) T.newGame();
            }

            auto listMoves = [&](const Position& p, MoveList& ml, int& term) {
                Position tmp = p;
                genLegal(tmp, pl.path, pl.mask, ml, term);
                };
            auto holds = [](const MoveList& ml, int mv) {
                for (int i = 0; i < ml.n; ++i) if (ml.m[i] == mv) return true;
                return false;
                };
            float lastEval = 0.5f, lastDepth = 0.0f;
            double lastSec = 0.0;
            // Candidates for the position currently on the board, best first.
            // Keeping the whole list means a destination the site turns down can
            // be replaced without searching again.
            vector<int> cand, candNext;
            // The long think belongs to the first position of the turn that
            // actually offers a choice: forced moves are played straight away
            // and must not eat it, and it is spent once per turn - not once per
            // attempt at the turn.
            if (wholeRoll) longUsed = false;
            auto chooseMove = [&](const Position& p, const Position& pLive,
                vector<int>& out, int& outTerm,
                const atomic<bool>* stopWith = nullptr) -> bool {
                    MoveList mlAll; int termAll = 0;
                    listMoves(p, mlAll, termAll);
                    if (mlAll.n == 0) return false;             // dice spent, turn over
                    MoveList mlLive = mlAll;
                    if (partial) { int t; listMoves(pLive, mlLive, t); }
                    outTerm = termAll;

                    out.clear();
                    for (int i = 0; i < mlAll.n; ++i)
                        if (!partial || holds(mlLive, mlAll.m[i])) out.push_back(mlAll.m[i]);
                    // The two readings can agree on nothing at all: with a lit
                    // rook and a full roll of knight-rook-knight, the one rook
                    // move is illegal under the full set and every knight move is
                    // illegal under the lit one. Falling back to the other
                    // position's moves then offers the site exactly what its dim
                    // faces already refused, and the turn loops forever. A dim
                    // face is the site's own answer, so keep to the moves the lit
                    // set allows.
                    if (out.empty()) for (int i = 0; i < mlAll.n; ++i) out.push_back(mlAll.m[i]);
                    // Nothing to think about when there is only one move.
                    if (out.size() == 1) { lastSec = 0.0; lastDepth = 0.0f; return true; }

                    // The turn's one real think is the first position with a
                    // choice. Anything after that only exists to fill the time
                    // the mouse spends placing the previous move, so it runs
                    // until that move is confirmed instead of a fixed second.
                    // One long think per roll. A position reread mid-turn gets
                    // only a short one, because the roll's think is spent.
                    const double sec = stopWith ? 600.0
                        : (longUsed ? min(2.0, firstSec) : firstSec);
                    longUsed = true;
                    const auto tSearch = steady_clock::now();

                    Position root = p;
                    vector<moveState> rm;
                    vector<int> pv;
                    mctsBatchedMT(T, root, pl.path, pl.mask, sec,
                        lastEval, lastDepth, rm, pv, 0, 0, autoSearchThreads(), true,
                        nullptr, &nn, /*stopOnWin=*/true, 0, 0.0, stopWith);
                    lastSec = duration<double>(steady_clock::now() - tSearch).count();

                    // Most-visited move first: the search returns its root moves
                    // reordered by the dif heuristic, which is a reporting aid,
                    // not the move to play.
                    vector<const moveState*> byVisits;
                    byVisits.reserve(rm.size());
                    for (const moveState& ms : rm) byVisits.push_back(&ms);
                    stable_sort(byVisits.begin(), byVisits.end(),
                        [](const moveState* a, const moveState* b) {
                            if (a->visits != b->visits) return a->visits > b->visits;
                            return a->eval > b->eval;
                        });
                    vector<int> order;
                    order.reserve(byVisits.size() + mlLive.n);
                    for (const moveState* ms : byVisits) order.push_back(ms->move);
                    for (int i = 0; i < mlLive.n; ++i) order.push_back(mlLive.m[i]);
                    (void)pv;

                    out.clear();
                    for (int mv : order) {
                        if (!holds(mlAll, mv)) continue;
                        if (partial && !holds(mlLive, mv)) continue;
                        if (find(out.begin(), out.end(), mv) != out.end()) continue;
                        out.push_back(mv);
                    }
                    if (out.empty()) for (int i = 0; i < mlAll.n; ++i) out.push_back(mlAll.m[i]);
                    return !out.empty();
                };

            bool broke = false;
            int term = 0, nextTerm = 0;
            int played = 0;               // moves this roll actually bought
            if (chooseMove(pos, posLive, cand, term)) {
                for (int step = 0; step < 24 && !cand.empty(); ++step) {
                    const int move = cand.front();
                    Position after = pos;
                    makeMove(after, pl.mask, move);
                    Position afterLive = posLive;
                    if (partial) makeMove(afterLive, pl.mask, move);

                    // A pawn moving diagonally onto an empty square is an en
                    // passant capture - worth calling out, because the right to
                    // make it is reconstructed from the screen and is the one
                    // part of the position that cannot be seen directly.
                    {
                        const int mf = move & 63, mt = (move >> 6) & 63;
                        if ((pos.piece[0] & bit(mf)) && ((mt - mf) & 7)
                            && !(pos.color[!pos.side] & bit(mt)))
                            cout << "[play] en passant capture\n";
                    }
                    const auto tStep = steady_clock::now();
                    cout << "[play] " << moveToStr(move) << "  eval " << fixed << setprecision(3)
                        << lastEval << " depth " << setprecision(1) << lastDepth
                        << " t " << lastSec
                        << " gap " << duration<double>(tStep - lastMoveAt).count() << '\n';
                    lastMoveAt = tStep;

                    // The mouse works while the next position is searched, so
                    // the engine never sits idle waiting for the site.
                    atomic<int> res{ -2 };
                    atomic<bool> placed{ false };
                    const Position beforeCopy = pos, afterCopy = after;
                    const int mv = move, fl = pl.flip;
                    thread mover([&res, &placed, beforeCopy, afterCopy, mv, fl] {
                        res.store(playMove(beforeCopy, afterCopy, mv, fl));
                        placed.store(true);
                        });

                    bool haveNext = false;
                    candNext.clear();
                    if (!term) haveNext = chooseMove(after, afterLive, candNext, nextTerm, &placed);
                    mover.join();

                    const int r = res.load();
                    if (r == 3) {
                        // Nothing about the position has changed - the site
                        // simply was not listening. Sit out the reconnect and
                        // play the very same move, no rereading, no reshuffling
                        // of candidates.
                        cout << "[play] connection lost, waiting for it to come back\n";
                        for (int waited = 0; waited < 600; ++waited) {
                            Sleep(1000);
                            if (!reconnecting(grabAll())) break;
                        }
                        cout << "[play] connection back, repeating " << moveToStr(move) << '\n';
                        lastAccepted = steady_clock::now();
                        Sleep(1500);
                        --step;              // a reconnect is not an attempt
                        continue;
                    }
                    if (r == 2) {
                        // The site does not offer this destination. Nothing has
                        // moved, so just drop the move and take the next
                        // candidate for the same position.
                        cout << "[play] " << moveToStr(move) << ": the site will not pick that piece up\n";
                        const int stuck = move & 63;
                        cand.erase(remove_if(cand.begin(), cand.end(),
                            [stuck](int m) { return (m & 63) == stuck; }), cand.end());
                        continue;
                    }
                    if (r != 1) {
                        if (r < 0) {
                            banned = move;
                            MoveList ml; int t;
                            listMoves(pos, ml, t);
                            array<int, 64> shown;
                            boardOfPos(pos, pl.flip, shown);
                            cout << "[play] refused position, dice " << diceIntToFen(pos.dice)
                                << (partial ? " live " + diceIntToFen(posLive.dice) : "")
                                << ", side " << (pos.side ? "black" : "white")
                                << ", castle " << pos.castle << ", step " << step << '\n';
                            dumpBoard(shown);
                            cout << "[play] legal:";
                            for (int i = 0; i < ml.n; ++i) cout << ' ' << moveToStr(ml.m[i]);
                            cout << '\n';
                        }
                        broke = true;
                        break;
                    }
                    banned = 0;
                    lastAccepted = steady_clock::now();
                    ++played;
                    pos = after;
                    posLive = afterLive;
                    if (term) {                           // king captured, game over
                        pl.outcome = 1;
                        broke = true;
                        break;
                    }
                    if (!haveNext) break;                 // dice spent, turn over
                    cand = candNext;
                    term = nextTerm;
                }
            }

            if (played) pl.series.push_back(played);

            if (broke) {
                // Our model of the position disagrees with the site - almost
                // always a misread die. Keep the colour (it cannot change
                // inside a game) and rebuild everything else from the screen.
                pl.haveLast = 0;
                ++rejects;
                if (rejects >= 3) {
                    // Either the site lost its connection or our reading is
                    // wrong in a way rereading will not fix; hammering it with
                    // clicks only burns the clock.
                    cout << "[play] " << rejects << " rejected turns in a row, waiting\n";
                    Sleep(min(30000, 3000 * rejects));
                }
                else Sleep(1200);
                continue;
            }
            rejects = 0;

            boardOfPos(pos, pl.flip, pl.lastBoard);
            pl.haveLast = 1;
            pl.castle = pos.castle;
            // The position our turn ends in is the one frame of the opponent's
            // turn we never have to read off the screen - we computed it. Make
            // it the baseline, so watching starts at the exact moment they
            // begin rather than whenever the next grab happens to land. Without
            // this the whole of a short turn can pass between two frames and a
            // double push goes unseen.
            pl.watchBoard = pl.lastBoard;
            pl.haveWatch = 1;
            // Just long enough for the last click to register. Every extra
            // millisecond here is time the opponent moves unwatched.
            Sleep(80);
        }
        catch (const std::exception& e) {
            // A run is meant to last for days; one bad iteration should cost a
            // turn, not the whole session. Rebuild from the screen and carry on.
            cout << "[play] recovered from error: " << e.what() << '\n';
            diagLogLine(std::string("[play] exception: ") + e.what());
            pl.flip = -1;
            pl.haveLast = 0;
            Sleep(2000);
        }
        catch (...) {
            cout << "[play] recovered from unknown error\n";
            diagLogLine("[play] unknown exception");
            pl.flip = -1;
            pl.haveLast = 0;
            Sleep(2000);
        }
    }

}   // namespace SP

int main() {
    // Unbuffered stdout: progress lines reach redirected log files immediately
    // (std::cout syncs with stdio, so this covers all engine output).
    setvbuf(stdout, nullptr, _IONBF, 0);
    SetProcessDPIAware();   // screen modes work in physical pixels
    // These handlers existed but were never switched on, so a crash during a
    // long unattended run left nothing behind to explain it.
    installCrashDiagnostics();
    try {
        const std::string ptFile = "net.pt";
        const std::string emaFile = "net_ema.pt";
        const std::string planFile = "net.plan";

        std::cout << "Enter FEN ('960' random Chess960, '-' Training, 'd' screen diagnostics,\n"
            "a whole number of seconds = play on screen with that much analysis after the roll):\n";
        std::string fen;
        std::getline(std::cin, fen);
        while (!fen.empty() && (fen.back() == '\r' || fen.back() == ' ')) fen.pop_back();

        // Screen diagnostics need no net: print the geometry and what is read.
        if (fen == "d") {
            SP::diagnose(120);
            return 0;
        }
        // A bare number selects the play-on-screen mode.
        bool playMode = !fen.empty() && fen.find_first_not_of("0123456789") == std::string::npos;
        double playSeconds = playMode ? atof(fen.c_str()) : 0.0;
        if (playMode && playSeconds < 1.0) playSeconds = 1.0;

        if (fen == "widen192") {
            // Reads "srcNet.pt srcEma.pt" from the next stdin line;
            // writes widened 10x192 net.pt and net_ema.pt into cwd.
            std::string srcPt = "net128.pt", srcEma = "net128_ema.pt";
            std::string cfgLine;
            if (std::getline(std::cin, cfgLine) && !cfgLine.empty()) {
                std::istringstream is(cfgLine);
                is >> srcPt >> srcEma;
            }
            const bool ok1 = createNet192FromFile(srcPt, "net.pt");
            const bool ok2 = createNet192FromFile(srcEma, "net_ema.pt");
            return (ok1 && ok2) ? 0 : 1;
        }
        if (fen == "-") {
            diagLogLine("[main] entering Training()");
            Training(INT_MAX); // infinite: runs until the process is force-stopped
            diagLogLine("[main] Training() finished normally");
            return 0;
        }

        Net model;
        Net emaModel;
        initAllOrExit(model, emaModel, ptFile, emaFile, planFile);
        if (!g_trtReady) {
            diagLogLine("[main] TensorRT engine not ready");
            std::cout << "TensorRT engine is not loaded.\n";
            return 1;
        }
        // Second TRT runner: pipelines CPU encode/expand with GPU compute in play modes.
        {
            std::lock_guard<std::mutex> lk(g_trtMutex);
            torch::NoGradGuard ng;
            if (g_trt2.initOrCreate(planFile) && trtRefitFromTorchModel(g_trt2, emaModel)) {
                g_trt2Ready = true;
            }
            else {
                g_trt2.shutdown();
                std::cout << "[TRT2] second runner unavailable; single-runner mode.\n";
            }
        }
        if (fen == "ab") {
            // Optional config line on stdin:
            //   "t1 r1 d1 o1 p1  t2 r2 d2 o2 p2  moveTimeMs"
            // t=threads, r=tree reuse, d=dual infer, o=old net (net_old.plan), p=persistent server.
            // Defaults: P1 = final (2 threads, reuse, dual, persistent), P2 = original.
            unsigned t1 = 2; int r1 = 1; int d1 = 1; int o1 = 0; int pp1 = 1;
            unsigned t2 = 1; int r2 = 0; int d2 = 0; int o2 = 0; int pp2 = 0;
            double moveMs = 400.0;
            int a1 = 0, a2 = 0;          // adaptive time management
            double bankSec = 0.0;        // per-game clock shared by both sides
            {
                std::string cfgLine;
                if (std::getline(std::cin, cfgLine) && !cfgLine.empty()) {
                    std::istringstream is(cfgLine);
                    is >> t1 >> r1 >> d1 >> o1 >> pp1 >> t2 >> r2 >> d2 >> o2 >> pp2 >> moveMs
                        >> a1 >> a2 >> bankSec;
                }
            }
            if (o1 || o2) {
                if (!ensureOldRunnerReady("net_old.plan")) {
                    std::cout << "[ab] net_old.plan not available; cannot run old-net side.\n";
                    return 1;
                }
            }
            AbPlayerCfg p1Cfg{ t1, r1 != 0, d1 != 0, o1 != 0, pp1 != 0 };
            AbPlayerCfg p2Cfg{ t2, r2 != 0, d2 != 0, o2 != 0, pp2 != 0 };
            p1Cfg.adaptive = a1; p1Cfg.bankSec = bankSec;
            p2Cfg.adaptive = a2; p2Cfg.bankSec = bankSec;
            MatchStatsGeneric st = runAbStrengthMatch(100000, moveMs / 1000.0, p1Cfg, p2Cfg);
            std::cout << "[ab] final W/L/D="
                << st.p1Wins << '/' << st.p2Wins << '/' << st.draws << std::endl;
            std::lock_guard<std::mutex> lk(g_trtMutex);
            g_trt.shutdown();
            g_trtReady = false;
            if (g_trt2Ready) { g_trt2.shutdown(); g_trt2Ready = false; }
            return 0;
        }
        if (fen == "arena") {
            // Net-vs-net gate at fixed sims: reads two plan paths from the next stdin line
            // (default: "net.plan net_old.plan").
            std::string n1 = "net.plan", n2 = "net_old.plan";
            {
                std::string cfgLine;
                if (std::getline(std::cin, cfgLine) && !cfgLine.empty()) {
                    std::istringstream is(cfgLine);
                    is >> n1 >> n2;
                }
            }
            arena(n1, n2);
            std::lock_guard<std::mutex> lk(g_trtMutex);
            g_trt.shutdown();
            g_trtReady = false;
            if (g_trt2Ready) { g_trt2.shutdown(); g_trt2Ready = false; }
            return 0;
        }
        if (fen == "difprobe") {
            int n = 60; double sec = 1.0;
            std::string cfgLine;
            if (std::getline(std::cin, cfgLine) && !cfgLine.empty()) {
                std::istringstream is(cfgLine);
                is >> n >> sec;
            }
            difProbe(n, sec);
            std::lock_guard<std::mutex> lk(g_trtMutex);
            g_trt.shutdown();
            g_trtReady = false;
            if (g_trt2Ready) { g_trt2.shutdown(); g_trt2Ready = false; }
            return 0;
        }
        if (fen == "match") {
            MatchStatsGeneric st = runTimedPvMatch(10000);
            std::cout << "[timed-pv-match] final p1/p2/draw="
                << st.p1Wins << '/' << st.p2Wins << '/' << st.draws
                << " p1Score=" << std::fixed << std::setprecision(4) << st.p1Score() << std::endl;
            std::lock_guard<std::mutex> lk(g_trtMutex);
            g_trt.shutdown();
            g_trtReady = false;
            if (g_trt2Ready) { g_trt2.shutdown(); g_trt2Ready = false; }
            return 0;
        }
        if (playMode) {
            SP::run(playSeconds);
            return 0;
        }
        if (fen == "s") {
            ROLL = 0;
            START(POS);
            START();
            thread loadThread(LOAD);
            thread searchThread(SEARCH);
            loadThread.join();
            searchThread.join();
        }
        Position pos;
        std::array<uint64_t, 4> path;
        std::array<int, 64> mask;

        if (fen == "960") chess960(pos, path, mask);
        else              fenToPositionPathMask(fen, pos, path, mask);



        MoveList ml;
        int term = 0;
        Position tmp = pos;


        float mctsEvalWhite = 0.5f;
        float mctsAvgDepth = 0.0f;
        std::vector<int> pvBeforeRoll;
        std::vector<moveState> rootMoves;
        const size_t nodePow2 = 1ull << 26;
        const size_t edgeCap = 1ull << 29;
        MCTSTable T(nodePow2, edgeCap);
        mctsBatchedMT(T, pos, path, mask, INT_MAX, mctsEvalWhite, mctsAvgDepth, rootMoves, pvBeforeRoll, 2, 0, autoSearchThreads(), true);
        clearConsoleFull();
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "depth=" << mctsAvgDepth << std::endl;
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "eval=" << mctsEvalWhite << std::endl;

        for (size_t i = 0; i < pvBeforeRoll.size(); ++i) {
            if (i) std::cout << ' ';
            std::cout << moveToStr(pvBeforeRoll[i]);
        }

        if (rootMoves.size()) {
            cout << endl;
            Dif(rootMoves[0].dif);
        }
        cout << endl;
        for (const auto& ms : rootMoves) {
            int d = (int)std::to_string(ms.visits).size();
            int spacesBeforePrior = 1 + (getMaxVisitsLen(rootMoves) - d);

            std::cout
                << moveToStr(ms.move)
                << " eval " << ms.eval
                << " visits " << ms.visits
                << std::string(spacesBeforePrior, ' ')
                << "prior " << ms.prior
                << ' ';
            Dif(ms.dif);
            cout << endl;
        }
        {
            std::lock_guard<std::mutex> lk(g_trtMutex);
            g_trt.shutdown();
            g_trtReady = false;
            if (g_trt2Ready) { g_trt2.shutdown(); g_trt2Ready = false; }
        }

        diagLogLine("[main] finished normally");
        cin.get();
        return 0;
    }
    catch (const std::exception& e) {
        diagLogLine(std::string("[main] fatal std::exception: ") + e.what());
        return 1;
    }
    catch (...) {
        diagLogLine("[main] fatal unknown exception");
        return 1;
    }
}
