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
    int   visits;
    float prior;
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
        cout << (r + 1) << " | ";
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

                uint64_t capLp = capL & RANK_8, capLn = capL & ~RANK_8;
                uint64_t capRp = capR & RANK_8, capRn = capR & ~RANK_8;

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

                uint64_t capLp = capL & RANK_1, capLn = capL & ~RANK_1;
                uint64_t capRp = capR & RANK_1, capRn = capR & ~RANK_1;

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

                uint64_t capLp = capL & RANK_8, capLn = capL & ~RANK_8;
                uint64_t capRp = capR & RANK_8, capRn = capR & ~RANK_8;

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

                uint64_t capLp = capL & RANK_1, capLn = capL & ~RANK_1;
                uint64_t capRp = capR & RANK_1, capRn = capR & ~RANK_1;

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

void makeRandom(Position& pos, TTNode* node) {
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
    pos.dice = Dice[Range(Random)];
    if (pawns) {
        if (pos.side == 0)dist = clz64(pawns) >> 3;   // MSVC-safe
        else dist = ctz64(pawns) >> 3;            // MSVC-safe
    }
    for (int i = 0; i < 5; i++)
        while (dicePiece[pos.dice][i] && (pos.color[pos.side] & pos.piece[i]) == 0 && dist > dicePiece[pos.dice][0])
            pos.dice = newDice[pos.dice][i];
    pos.key ^= ZDice[pos.dice];
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
// Чтобы (base + step * k) обходил все residue-классы, step должен быть взаимно прост с 216,
// т.е. НЕ делиться ни на 2, ни на 3.
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
    // fallback: если node нет, используем старое случайное поведение
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
static constexpr int NET_CHANNELS = 128;
static constexpr int POLICY_P = 73;
static constexpr double AI_BN_EPS = 1e-5;

// SE (affine)
static constexpr int SE_CHANNELS = 16;   // для C=128 обычно 8..16; 16 сильнее

// Heads
static constexpr int HEAD_POLICY_C = 32; // 32 для 10x128 — стандартный хороший выбор
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

            // по желанию на время отладки:
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

    // ВАЖНО: никаких sleep_for(microseconds) — на многих ОС это вырождается в ~1ms.
    // Yield делаем редко, чтобы не терять throughput.
    if (spins == 256 || spins == 1024 || spins == 4096) {
        std::this_thread::yield();
    }
    if (spins > 16384) {
        // если очень долго — начинаем yield чаще, но всё равно без сна
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
                // rare tag collision -> probe дальше
            }

            if (mt == TAG_EMPTY32) return nullptr;

            idx = (idx + 1) & mask;
            ++probe;
        }

        return nullptr;
    }
};

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

// Выбор PV: сначала max visits, затем max Q, затем max prior.
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
    float c_base = 19652.0f;
};

static const SearchParams kDefaultSearchParams{};

static AI_FORCEINLINE float cpuctFromVisits(
    uint32_t parentVisits,
    bool isRoot,
    const SearchParams& sp) {
    float c = sp.c_init + std::log(((float)parentVisits + sp.c_base + 1.0f) / sp.c_base);
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
static constexpr uint32_t VLOSS_N = 1;     // обычно 1; 2-3 имеет смысл только при очень многих потоках
static constexpr float    VLOSS_VALUE = 0.0f; // value в шкале [0..1]; 0.0 = "loss for side-to-move"
static constexpr bool     VLOSS_BUMP_NODE_VISITS = false; // опционально

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

static void waitGroupWaitZero(SearchWaitGroup* wg) {
    if (!wg) return;
    std::unique_lock<std::mutex> lk(wg->m);
    wg->cv.wait(lk, [&] {
        return wg->pending.load(std::memory_order_acquire) == 0;
        });
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
        // Должно быть невозможно, но если mapping сломался —
        // даём почти -inf, чтобы softmax дал ~0.
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

static AI_FORCEINLINE void applyVirtualLoss(TraceStep& s) {
    if (!s.vloss) return;

    if (VLOSS_BUMP_NODE_VISITS && s.node) {
        s.node->visits.fetch_add(VLOSS_N, std::memory_order_relaxed);
        // valueSum узла НЕ трогаем (классика)
    }

    if (s.edge) {
        s.edge->visits.fetch_add(VLOSS_N, std::memory_order_relaxed);
        // “loss” в [0..1] шкале => добавляем W как будто вернулся VLOSS_VALUE
        if (VLOSS_VALUE != 0.0f) {
            atomicAddDouble(s.edge->valueSum, (double)VLOSS_VALUE * (double)VLOSS_N);
        }
        // если VLOSS_VALUE=0.0f, valueSum можно не трогать вообще
    }
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

static AI_FORCEINLINE void backprop(TTNode* leaf, float v, Trace& tr) {
    rollbackVirtualLoss(tr);

    leaf->addVisitAndValue(v);

    for (int i = tr.n - 1; i >= 0; --i) {
        TraceStep& s = tr.st[i];
        if (s.flip) v = 1.0f - v;
        if (s.edge) s.edge->addVisitAndValue(v);
        s.node->addVisitAndValue(v);
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

        backprop(p.leaf, v, p.trace);
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

        backprop(p.leaf, v, p.trace);
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

    backprop(p.leaf, v, p.trace);
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

        backprop(p.leaf, v, p.trace);
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

        backprop(p.leaf, v, p.trace);
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

    backprop(p.leaf, v, p.trace);
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

    bool busyFlag = false; // protected by m

    std::vector<float> neutralPol;     // [POLICY_SIZE]
    std::vector<float> neutralLogits;  // [AI_MAX_MOVES]

    explicit InferenceServer(MCTSTable& tab) : T(tab) {
        q.clear();
        neutralPol.assign((size_t)POLICY_SIZE, 0.0f);
        neutralLogits.assign((size_t)AI_MAX_MOVES, 0.0f);
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

    void stopAndDrain() {
        {
            std::lock_guard<std::mutex> lk(m);
            stop.store(true, std::memory_order_relaxed);
        }
        cvNotEmpty.notify_all();
        cvNotFull.notify_all();
        cvIdle.notify_all();

        if (th.joinable()) th.join();
    }

    int size() const {
        return qSize.load(std::memory_order_relaxed);
    }

    void waitIdle() {
        std::unique_lock<std::mutex> lk(m);
        cvIdle.wait(lk, [&] {
            return q.empty() && !busyFlag;
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

    void run() {
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
            bool ok = g_trt.inferBatchGather(batchPtrs.data(), B);
            for (int i = 0; i < B; ++i) {
                float v = ok ? g_trt.valueHost(i) : 0.5f;
                const float* logits = ok ? g_trt.gatherLogitsHostPtr(i) : neutralLogits.data();
                expandLeafWithGatheredLogits(T, *jobs[(size_t)i], v, logits);
            }
#else
            std::vector<Position> posBatch((size_t)B);
            for (int i = 0; i < B; ++i) posBatch[(size_t)i] = jobs[(size_t)i]->pos;

            bool ok = g_trt.inferBatch(posBatch.data(), B);
            for (int i = 0; i < B; ++i) {
                float v = ok ? g_trt.valueHost(i) : 0.5f;
                const float* pol = ok ? g_trt.policyHostPtr(i) : neutralPol.data();
                expandLeafWithOutputs(T, *jobs[(size_t)i], v, pol);
            }
#endif
            };

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
        }

        {
            std::lock_guard<std::mutex> lk(m);
            busyFlag = false;
            qSize.store((int)q.size(), std::memory_order_relaxed);
            if (q.empty()) cvIdle.notify_all();
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
    int term) {
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
        publishReady(root, rootPos.key, 0, 0, 1, 0);
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
    if (!g_trt.inferBatchGather(&p, 1)) {
        v = 0.5f;
        std::vector<float> z((size_t)AI_MAX_MOVES, 0.0f);
        expandLeafWithGatheredLogits(T, p, v, z.data());
        pGuard.release();
        return;
    }
    v = g_trt.valueHost(0);
    const float* logits = g_trt.gatherLogitsHostPtr(0);
    expandLeafWithGatheredLogits(T, p, v, logits);
    pGuard.release();
#else
    std::vector<float> pol((size_t)POLICY_SIZE, 0.0f);
    if (!g_trt.inferBatch(&p.pos, 1, &v, pol.data())) {
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

    if (AI_UNLIKELY(T.abort.load(std::memory_order_relaxed))) return false;

    Position pos = rootPos;
    Trace tr; tr.reset();
    bool isRoot = true;

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

                backprop(node, 1.0f, tr);
                publishReady(node, pos.key, 0, 0, 1, 0);
                claim.release();
                if (diag) diag->depth = (uint32_t)tr.n;
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
            if (diag) diag->depth = (uint32_t)tr.n;
            return true;
        }

        // Expanded
        if (node->terminal) {
            backprop(node, 1.0f, tr);
            if (diag) diag->depth = (uint32_t)tr.n;
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
                backprop(node, vLeaf, tr);
                if (diag) diag->depth = (uint32_t)tr.n;
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
        applyVirtualLoss(step);

        makeMove(pos, mask, e->move);
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
static void extractBestPVUntilChance(MCTSTable& T,
    const Position& rootPos,
    const std::array<int, 64>& mask,
    std::vector<int>& outPV,
    int maxDepth = 256) {
    outPV.clear();

    Position pos = rootPos;

    for (int depth = 0; depth < maxDepth; ++depth) {
        TTNode* n = T.findNodeNoInsert(pos.key);
        if (!n) break;

        uint8_t ex = n->expanded.load(std::memory_order_acquire);
        if (ex != 1) break;

        if (n->terminal) break;

        // chance-узел: остановиться ДО makeRandom()
        if (n->chance) break;

        if (n->edgeCount == 0) break;

        TTEdge* e0 = T.edgePtr(n->edgeBegin);
        int bi = selectBestPVEdge(*n, e0);
        int m = e0[bi].move;

        outPV.push_back(m);
        makeMove(pos, mask, m);
    }
}
Position POS;
array<uint64_t,4> PATH;
array<int,64> MASK;
void mctsBatchedMT(Position& rootPos,
    std::array<uint64_t, 4>& path,
    std::array<int, 64>& mask,
    double timeSec,
    float& outEvalWhite,
    std::vector<moveState>& outRootMoves,
    std::vector<int>& outPVBeforeRoll,
    int write,
    int abort) {
    MoveList ml;
    int term;
    genLegal(rootPos, path, mask, ml, term);

    outPVBeforeRoll.clear();

    if (term) {
        outEvalWhite = 1 - rootPos.side;
        outRootMoves.clear();
        outRootMoves.push_back({ ml.m[0], outEvalWhite, 0 });
        outPVBeforeRoll.push_back(ml.m[0]);
        if (write == 1) {
            clearConsoleFull();
std::cout << moveToStr(ml.m[0]) << std::endl;
        }
        return;
    }

    const size_t nodePow2 = 1ull << 21;
    const size_t edgeCap = 1ull << 25;

    MCTSTable T(nodePow2, edgeCap);

    TTNode* rootNode = T.getNode(rootPos.key);
    if (!rootNode) {
        outEvalWhite = 0.5f;
        outRootMoves.clear();
        outPVBeforeRoll.clear();
        return;
    }

    ensureRootExpanded(T, rootPos, path, mask, ml, term);

    if (T.abort.load(std::memory_order_acquire)) {
        outEvalWhite = 0.5f;
        outRootMoves.clear();
        outPVBeforeRoll.clear();
        return;
    }



    InferenceServer nnServer(T);
    nnServer.start();
    InferenceServerStopGuard nnServerGuard(nnServer);

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const unsigned threads = std::max(1u, hw / 2);

    const auto t0 = std::chrono::steady_clock::now();
    const auto tEnd = t0 + std::chrono::duration<double>(timeSec);
    auto tNextWrite = t0 + std::chrono::seconds(1);
    auto tNextAbortCheck = t0 + std::chrono::milliseconds(50);

    std::atomic<bool> stop{ false };
    AtomicStopGuard stopGuard(stop);

    std::atomic<uint64_t> simOK{ 0 }, simFail{ 0 }, nnExp{ 0 };

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

                bool ok = runOneSim(T, rootPos, path, mask,
                    localPending, needNN,
                    jitterBase + (uint32_t)k * 1337u,
                    kDefaultSearchParams);

                if (!ok) {
                    simFail.fetch_add(1, std::memory_order_relaxed);
                    if (T.abort.load(std::memory_order_relaxed)) break;
                    continue;
                }

                didUsefulWork = true;
                simOK.fetch_add(1, std::memory_order_relaxed);

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
                rootMovesNow.push_back(moveState{ e.move, ev, (int)v, p });
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
        }

        std::vector<int> pvNow;
        extractBestPVUntilChance(T, rootPos, mask, pvNow, 256);

        clearConsoleFull();
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "eval=" << mctsEvalWhiteNow << '\n';
        for (size_t i = 0; i < pvNow.size(); ++i) {
            if (i) std::cout << ' ';
            std::cout << moveToStr(pvNow[i]);
        }
        std::cout << '\n';
        for (const auto& ms : rootMovesNow) {
            int d = (int)std::to_string(ms.visits).size();
            int spacesBeforePrior = 1 + (to_string(rootMovesNow[0].visits).size() - d);

            std::cout
                << moveToStr(ms.move)
                << " eval " << ms.eval
                << " visits " << ms.visits
                << std::string(spacesBeforePrior, ' ')
                << "prior " << ms.prior
                << '\n';
        }
        std::cout.flush();
    };

    bool forceExit = false;
    while (std::chrono::steady_clock::now() < tEnd) {
        if (T.abort.load(std::memory_order_relaxed)) break;
        auto now = std::chrono::steady_clock::now();

        if (abort && POS.key != rootPos.key && now >= tNextAbortCheck) {
            forceExit = true;
            break;
        }
        while (now >= tNextAbortCheck) {
            tNextAbortCheck += std::chrono::milliseconds(50);
        }

        if (write == 1) {
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

    nnServer.stopAndDrain();
    nnServerGuard.release();

    stopGuard.release();

    float qRoot = nodeQ(*rootNode);
    outEvalWhite = (rootPos.side == 0) ? qRoot : (1.0f - qRoot);

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

            outRootMoves.push_back(moveState{ e.move, ev, (int)v, p });
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
    }

    // NEW: вытаскиваем первую линию до броска кубиков
    extractBestPVUntilChance(T, rootPos, mask, outPVBeforeRoll, 256);

    (void)simOK; (void)simFail; (void)nnExp;
    if (forceExit) return;
}

// ===================== TRAINING PATCH BEGIN (FINAL) =====================
// (продолжение будет в сообщении 2/2)
// ===================== TRAINING PATCH BEGIN (FINAL) =====================
// ВСТАВЬ ЭТО ВМЕСТО ТВОЕГО ТЕКУЩЕГО `static void init()` И `int main()`
// (т.е. удалить/заменить всё от `static void init()` до конца файла).



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

    explicit ResBlockImpl(int channels) : C(channels) {
        conv1 = register_module("conv1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(C, C, 3).padding(1).bias(false)));
        bn1 = register_module("bn1",
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(C).eps(AI_BN_EPS)));

        conv2 = register_module("conv2",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(C, C, 3).padding(1).bias(false)));
        bn2 = register_module("bn2",
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(C).eps(AI_BN_EPS)));

        se = register_module("se", SEAffine(C, SE_CHANNELS));
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

        // Получаем сырые логиты
        v = valFC2->forward(v);


        if (!is_training()) {
            v = torch::sigmoid(v);
        }

        return { pol, v };
    }
};
TORCH_MODULE(Net);

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

    // Степень "свежести" данных (Prioritized Replay Lite).
    // 1.0  = полностью равномерный выбор (как было).
    // 0.75 = легкий приоритет свежим играм (золотая середина для AlphaZero).
    // 0.5  = сильный перекос в сторону только что сыгранных партий.
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
};

// ------------------------------------------------------------
// TRT refit из libtorch модели + пересоздание Context + CUDA Graph
// ------------------------------------------------------------

static std::mutex g_trtMutex;     // защищаем TRT enqueue/refit/serialize
static std::mutex g_modelMutex;   // защищаем чтение/запись весов модели и optimizer step
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

// Pretty-print missing refit weights (IMPORTANT: иначе refit может "молча" быть частичным).
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

// RAII: временно перевести модель в eval() на время refit и вернуть режим обратно.
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

    // IMPORTANT: refit делаем из eval(), чтобы BN running stats не менялись.
    ScopedModelEval evalGuard(model);
    torch::NoGradGuard ng;

    // Если модель на CUDA — можно синхронизироваться (опционально, но безопасно).
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
    // std::deque гарантирует стабильность адресов элементов.
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
// Inference server для обучения (CV вместо busy-wait), + g_trtMutex
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
// SearchPool: постоянные MCTS-воркеры (НЕ пересоздаём потоки на каждый search)
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

        // failFast() должен бросать только из управляющего потока.
        // Если кто-то случайно вызовет его из worker thread, просто не бросаем:
        // worker должен завершиться, а управляющий поток увидит fatal и сам бросит.
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

        waitGroupWaitZero(&wg);

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
// Dirichlet noise применяется ТОЛЬКО временно на root (не портит priors в TT навсегда)
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

    // Надёжно дожидаемся/захватываем expansion root-а.
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
            continue; // перечитать состояние root
        }

        uint8_t expected = 0;
        if (root->expanded.compare_exchange_strong(expected, 2,
            std::memory_order_acq_rel,
            std::memory_order_relaxed)) {
            break; // мы захватили expansion
        }

        // кто-то успел изменить expanded — пробуем снова
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
        publishReady(root, rootPos.key, 0, 0, 1, 0);
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

        outMoves.push_back(moveState{ e.move, ev, (int)v, e.prior() });
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
        const double v = (double)std::max(0, mv[i].visits);
        sum += std::pow(v + 1e-9, invTemp);
    }

    if (!(sum > 0.0)) return mv[0].move;

    std::uniform_real_distribution<double> d(0.0, sum);
    double r = d(Random);

    double acc = 0.0;
    for (size_t i = 0; i < mv.size(); ++i) {
        const double v = (double)std::max(0, mv[i].visits);
        acc += std::pow(v + 1e-9, invTemp);
        if (r <= acc) return mv[i].move;
    }

    return mv.back().move;
}

// policy target — SPARSE (idx/prob), idx в CHW: k=pl*64+sq
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
        sum += (double)std::max(0, mv[(size_t)i].visits);
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
        float p = (float)std::max(0, mv[(size_t)i].visits) * inv;

        outIdx[(size_t)i] = (uint16_t)k;
        outProbQ[(size_t)i] = quantizeProbU16(p);
    }
}

// временно (на один search) зашумливаем root priors и потом откатываем назад
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
// Self-play: переиспользуем один MCTSTable + один InferenceServerTrain + SearchPool
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
    SearchMovesFn&& searchMovesForSide)
{
    Position pos = startPos;

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
            makeRandom(pos, n);
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
    std::atomic<int> donePairs{ 0 };

    std::atomic<int> p1Wins{ 0 };
    std::atomic<int> p2Wins{ 0 };
    std::atomic<int> draws{ 0 };

    std::atomic<bool> abortAll{ false };

    std::mutex exM;
    std::exception_ptr ex;

    std::mutex printM;
    std::vector<std::thread> outer;
    outer.reserve(lanes.size());

    auto addResult = [&](int r) {
        if (r > 0) p1Wins.fetch_add(1, std::memory_order_relaxed);
        else if (r < 0) p2Wins.fetch_add(1, std::memory_order_relaxed);
        else draws.fetch_add(1, std::memory_order_relaxed);
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

                    lane.resetForNewGame();
                    addResult(playOneOnLane(lane, startPos, path, mask, /*p1IsWhite=*/true));

                    if (abortAll.load(std::memory_order_relaxed)) break;

                    lane.resetForNewGame();
                    addResult(playOneOnLane(lane, startPos, path, mask, /*p1IsWhite=*/false));

                    const int dp = donePairs.fetch_add(1, std::memory_order_relaxed) + 1;

                    if (progressEveryPairs > 0 && (dp % progressEveryPairs) == 0) {
                        MatchStatsGeneric snap;
                        snap.p1Wins = p1Wins.load(std::memory_order_relaxed);
                        snap.p2Wins = p2Wins.load(std::memory_order_relaxed);
                        snap.draws = draws.load(std::memory_order_relaxed);

                        std::lock_guard<std::mutex> lk(printM);
                        onProgress(dp * 2, snap);
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

    if ((games & 1) != 0 && !lanes.empty()) {
        Position startPos;
        std::array<uint64_t, 4> path;
        std::array<int, 64> mask;
        chess960(startPos, path, mask);

        Lane& lane = *lanes[0];
        lane.resetForNewGame();
        addResult(playOneOnLane(lane, startPos, path, mask, /*p1IsWhite=*/true));

        if (progressEveryPairs > 0) {
            MatchStatsGeneric snap;
            snap.p1Wins = p1Wins.load(std::memory_order_relaxed);
            snap.p2Wins = p2Wins.load(std::memory_order_relaxed);
            snap.draws = draws.load(std::memory_order_relaxed);

            onProgress(games, snap);
        }
    }

    out.p1Wins = p1Wins.load(std::memory_order_relaxed);
    out.p2Wins = p2Wins.load(std::memory_order_relaxed);
    out.draws = draws.load(std::memory_order_relaxed);
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
    int maxPlies = 256)
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
        findChanceNode, searchMoves);
}

static double computeLOSPercent(int wins, int losses);

static ArenaStats runArenaMatch(int games, int simsPerPos) {
    ArenaStats st;
    if (games <= 0) return st;

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());

    // В arena каждая lane = 2 SelfPlayContext, у каждого свой server + pool.
    // Поэтому держим threadsPerSide небольшим.
    const unsigned threadsPerSide = (hw >= 24 ? 2u : 1u);
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
        if ((playedGames % 100) == 0) {
            const double los = computeLOSPercent(s.p1Wins, s.p2Wins);
            std::cout << "[arena] games " << playedGames << "/" << games
                << "  W/L = " << s.p1Wins << "/" << s.p2Wins
                << "  score=" << s.p1Score()
                << "  LOS=" << std::fixed << std::setprecision(2) << los << "%\n";
        }
    };

    MatchStatsGeneric g = runUniversalMatchEngine(
        lanes,
        games,
        [&](ArenaLane& lane,
            const Position& startPos,
            const std::array<uint64_t, 4>& path,
            const std::array<int, 64>& mask,
            bool p1IsWhite) -> int {
            return playOneArenaGameOnLane(
                lane, startPos, path, mask, p1IsWhite, simsPerPos, 256);
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
        << " score1=" << std::fixed << std::setprecision(4) << score1
        << " LOS=" << std::setprecision(2) << los << "%\n";
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
    int maxPlies = 256)
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
        findChanceNode, searchMoves);
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
    int maxPlies = 256)
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
        findChanceNode, searchMoves);
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

    const SearchParams n1Params{ 0.70f, 0.16f, 19652.0f };
    const SearchParams n2Params{ 0.70f, 0.16f, 19652.0f };

    static constexpr int TOTAL_GAMES = 10000;
    static constexpr int SIMS_PER_POS = 800;
    static constexpr int MAX_PLIES = 256;

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const unsigned threadsPerSide = (hw >= 16 ? 2u : 1u);
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
        << " threads_per_side=" << threadsPerSide << "\n";

    auto onProgress = [&](int playedGames, const MatchStatsGeneric& s) {
        if ((playedGames % 100) == 0) {
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
            bool n1IsWhite) -> int {
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
                MAX_PLIES
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

    const SearchParams p1{ c_init1, fpu_reduction1, 19652.0f };
    const SearchParams p2{ c_init2, fpu_reduction2, 19652.0f };

    static constexpr int TOTAL_GAMES = 10000;
    static constexpr int SIMS_PER_POS = 800;
    static constexpr int MAX_PLIES = 256;

    BackendBinding backend{ g_trt, g_trtMutex };
    SharedInferenceServerTrain sharedSrv(backend);

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());

    // Tune lane легче arena: только 2 GameContext, shared NN server один на всех.
    const unsigned threadsPerSide = (hw >= 16 ? 2u : 1u);
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
        if ((playedGames % 100) == 0) {
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
            bool p1IsWhite) -> int {
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
                MAX_PLIES
            );
        },
        onProgress,
        /*progressEveryPairs=*/50);

    printTuneProgress(TOTAL_GAMES, g.p1Wins, g.p2Wins);
    std::cout << "[tune] finished\n";
}
static float lambdaQ=1;//ok
static float lambdaD=1;//ok
static float lambdaC=0.9;//ok
static float lambdaT=1;//ok
static float lambdaS=0;//ok
static float lambdaZ=0;//ok
static AI_FORCEINLINE float valueToSidePerspective(float v, int fromSide, int toSide) {
    v = clamp01(v);
    return (fromSide == toSide) ? v : (1.0f - v);
}

static AI_FORCEINLINE float chanceStepDecay(uint8_t chanceCount) {
if(chanceCount)return lambdaC;
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
        float v=lambdaQ;
        float sumV = v;
        float weighted = v*clamp01(game[(size_t)i].q);

        for (int j = i + 1; j < n; ++j) {
            v *= chanceStepDecay(chanceToNext[(size_t)j - 1]);
            sumV += v;

            const int sideJ = sideAtSample[(size_t)j];
            const float qInCurPerspective =
                valueToSidePerspective(game[(size_t)j].q, sideJ, sideCur);
            weighted += v * qInCurPerspective;
        }

        v=v*chanceStepDecay(chanceToNext[(size_t)n - 1])*lambdaT+sumV*lambdaS+lambdaZ;
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

    // Взвешенная цель по q с учетом количества chance-переходов между samples.
    buildChanceWeightedTargets(game, sideAtSample, chanceToNext, zWhite);

    rb.pushMany(game);
    outSamplesAdded += (int)game.size();
}

// ------------------------------------------------------------
// Trainer thread: sparse policy loss через gather(logp, idx)
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

    // Первичный refit
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

static AI_FORCEINLINE void waitForNoInferenceInFlight() {
    while (g_inferInFlight.load(std::memory_order_acquire) != 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}

static void safeRefitBarrierShared(SharedInferenceServerTrain& srv) {
    srv.waitIdle();
    waitForNoInferenceInFlight();
    srv.clearQueueUnsafeWhenIdle();
    waitForNoInferenceInFlight();
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
                    pliesDone.fetch_add((uint64_t)std::max(0, plyCount), std::memory_order_relaxed);
                    samplesDone.fetch_add((uint64_t)std::max(0, samplesAdded), std::memory_order_relaxed);

                    if (!terminated && plyCount >= maxPlies) {
                        truncatedDone.fetch_add(1, std::memory_order_relaxed);
                    }

                    if (sp.T.abort.load(std::memory_order_relaxed)) {
                        std::cerr << "[selfplay] game_ctx=" << i
                            << " aborted: oomCode="
                            << sp.T.oomCode.load(std::memory_order_relaxed)
                            << " -> reset table\n";
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

    // Trainer: инициализируем с нулевого шага, затем пытаемся восстановить optimizer
    Trainer trainer;

    trainer.init(model, emaModel);

    if (loadOptimizerState(optFile, trainer)) {
        std::cerr << "optimizer state restored.\n";

        // После restore optimizer ещё раз принудительно выставим LR по scheduler'у
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

    // Для training держим ровно 1 search-thread на игру.
    // Тогда лучший throughput обычно даёт больше параллельных игр,
    // но без безумного раздувания числа GameContext.
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

    std::cout << "[selfplay] parallel_games=" << PARALLEL_GAMES<< "\n";

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
                safeRefitBarrierShared(sharedSrv);

                // use old function as fixed-step runner by giving it a huge time budget
                didTrain = trainer.trainBlockBudgetMs(rb, model, emaModel,
                    /*budgetMs=*/24 * 60 * 60 * 1000,
                    /*maxStepsHard=*/targetSteps,
                    TRAIN_WARMUP_BATCHES);

                trainSampleCredits -= (double)didTrain * (double)trainer.B;
                if (trainSampleCredits < 0.0) trainSampleCredits = 0.0;

            }
        }

        // ===========================
        // 3) REFIT TRT
        // ===========================
        if (didTrain > 0) {
            safeRefitBarrierShared(sharedSrv);

            std::scoped_lock lk(g_modelMutex, g_trtMutex);
            torch::NoGradGuard ng;

            if (!trtRefitFromTorchModel(g_trt, emaModel)) {
                std::cerr << "[refit] TRT refit failed.\n";
            }
            else {
                ++refits;
            }
        }

        while (games >= nextArenaAt) {
            // Полностью ставим основной self-play на паузу на время арены:
            // убираем лишние worker threads / inference server activity.
            if (spRunning) {
                safeRefitBarrierShared(sharedSrv);
                for (auto& g : gamesCtx) {
                    if (g) g->stop();
                }
                spRunning = false;
            }

            bool arenaOk = true;

            // current TRT должен точно соответствовать текущему EMA model
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
                std::cout << "\n[arena] start: current vs old, games=2000, sims=800, triggerGames="
                    << nextArenaAt << "\n";

                ArenaStats ar = runArenaMatch(/*games=*/2000, /*simsPerPos=*/800);

                std::cout << "[arena] done: W/L = "
                    << ar.curWins << "/" << ar.oldWins
                    << "  score=" << ar.currentScore() << "\n";

                // promotion rule: если current > old, old := current
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

            // ВАЖНО: всегда поднимаем основной self-play обратно после арены.
            for (auto& g : gamesCtx) {
                if (g) g->start(SEARCH_THREADS_PER_GAME);
            }
            spRunning = true;
            prevSearchStats = snapshotAllSearchStats(gamesCtx);
            statWindowStart = std::chrono::steady_clock::now();

            // Даже если arena setup failed, не зацикливаемся на том же пороге.
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

            std::cout << "[autosave] Progress: " << games << " / " << targetGames << " games.\n";
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
float b=stof(fmtFixed(getAverageInferBatchSize(), 2));
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
<< " | Speed: " << stof(fmtFixed(nnCallsPerSec, 1))*b
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
            // не фатально: файла могло не быть
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
array<uint64_t,2> ep;
int castle;
vector<int> sqKey={2324,5950000,9121555,10242252,14571640,10172920,35020000,55080122,34360208,37770000,38410393,31600651,0};
vector<int> diceKey={1861,140516,7681148,6141254,10991103,7942210};
int NUMBER(vector<int>& v,int key){
int min,i,dist,num;
min=INT_MAX;
for(i=0;i<v.size();i++){
dist=abs(v[i]%10000-key%10000)+abs(v[i]/10000-key/10000);
if(dist<min){
num=i;
min=dist;
}
}
return num;
}
vector<int> S(int x1,int x2,int y1,int y2){
int w,h;
vector<int> s;
HDC d,m;
HBITMAP b;
BITMAPINFO p;
w=x2-x1+1;
h=y2-y1+1;
s.resize(w*h);
d=GetDC(0);
m=CreateCompatibleDC(d);
b=CreateCompatibleBitmap(d,w,h);
p={40,w,-h,1,32};
SelectObject(m,b);
BitBlt(m,0,0,w,h,d,x1,y1,13369376);
GetDIBits(d,b,0,h,s.data(),&p,0);
DeleteObject(b);
DeleteObject(m);
DeleteObject(d);
return s;
}
vector<int> S(){return S(0,1919,0,2399);}
int STATE(vector<int>& s){
int pixel1,pixel2,pixel3;
if(s.empty())return -1;
pixel1=s[960+1920*629];
pixel2=s[1335+1920*997];
pixel3=s[1442+1920*1955];
if(pixel1==-15455703||pixel2==-1||pixel3!=-5532810&&pixel3!=-11842499)return -1;
return pixel3==-11842499;
}
int FLIP(vector<int>& s){return s[417+1920*1752]==-665935;}
int SIDE(vector<int>& s){return STATE(s)!=FLIP(s);}
array<int,64> BOARD(vector<int>& s){
int sq,key,x,y,pixel;
array<int,64> board;
for(sq=0;sq<64;sq++){
if(FLIP(s)==0)sq^=56;else sq^=7;
key=0;
for(x=0;x<138;x++)for(y=0;y<138;y++){
pixel=s[407+138*(sq%8)+x+1920*(762+138*(sq/8)+y)];
key+=(pixel==-1)+10000*(pixel==-16777216);
}
board[sq]=NUMBER(sqKey,key);
}
return board;
}
void BOARD(array<int,64>& board,Position& pos){
int sq,piece;
pos.color={0,0};
pos.piece={0,0,0,0,0,0};
for(sq=0;sq<64;sq++){
piece=board[sq];
if(piece==12)continue;
pos.color[piece/6]|=bit(sq);
pos.piece[piece%6]|=bit(sq);
}
}
vector<int> SQUARE(array<int,64>& board1,array<int,64>& board2){
int sq;
vector<int> square;
for(sq=0;sq<64;sq++)if(board2[sq]!=board1[sq])square.push_back(sq);
return square;
}
int DICE(vector<int>& s,Position& pos){
int light,i,white,black,x,y,pixel,dice,dist;
uint64_t pawns;
string t;
vector<int> v;
light=0;
for(i=0;i<3;i++){
white=black=0;
for(x=0;x<158;x++)for(y=0;y<158;y++){
pixel=s[655+227*i+x+1920*(550+y)];
white+=pixel==-1||pixel==-8421505;
black+=pixel==-16777216;
light+=pixel==-1;
}
v.push_back(NUMBER(diceKey,max(white,black)+10000*min(white,black)));
}
if(light==0)return 0;
sort(v.begin(),v.end());
for(i=0;i<3;i++)t+=pieceChar(v[i]);
dice=diceFenToInt(t);
pawns=pos.color[pos.side]&pos.piece[0];
dist=6;
if(pawns)if(pos.side==0)dist=clz64(pawns)>>3;else dist=ctz64(pawns)>>3;
for(i=0;i<5;i++)while(dicePiece[dice][i]&&(pos.color[pos.side]&pos.piece[i])==0&&dist>dicePiece[dice][0])dice=newDice[dice][i];
return dice;
}
int EQUAL(vector<int>& s1,vector<int>& s2){
int i,j,k,n;
if(STATE(s2)!=STATE(s1))return 0;
if(STATE(s2)==-1)return 1;
for(i=0;i<3;i++)for(j=0;j<158;j++)for(k=0;k<158;k++){
n=655+227*i+j+1920*(550+k);
if(s2[n]!=s1[n])return 0;
}
return 1;
}
vector<int> NEW(vector<int> s1){
time_point<steady_clock> t1,t2;
vector<int> s2;
t1=steady_clock::now()+hours(1);
while(1){
t2=steady_clock::now();
s2=S();
if(EQUAL(s1,s2)==0){
t1=t2;
s1=s2;
continue;
}
if((t2-t1).count()>=100000000)return s2;
}
}
void START(Position& pos,array<uint64_t,4>& path,array<int,64>& mask){
POS.ep1={0,0};
POS.ep2=0;
pos.rook={0,7,56,63};
POS.castle=15;
POS.dice=0;
POS.key=0;
path={bit(1)|bit(2)|bit(3),bit(5)|bit(6),bit(57)|bit(58)|bit(59),bit(61)|bit(62)};
mask.fill(0);
mask[0]=1;
mask[4]=3;
mask[7]=2;
mask[56]=4;
mask[60]=12;
mask[63]=8;
}
void SET(vector<int>& s,Position& pos,array<uint64_t,4>& path,array<int,64>& mask){
int sq,piece;
array<int,64> board;
board=BOARD(s);
pos.color={0,0};
pos.piece={0,0,0,0,0,0};
for(sq=0;sq<64;sq++){
piece=board[sq];
if(piece==12)continue;
pos.color[piece/6]|=bit(sq);
pos.piece[piece%6]|=bit(sq);
}
pos.side=SIDE(s);
pos.ep1={0,0};
pos.ep2=0;
pos.rook={0,7,56,63};
pos.castle=0;
pos.dice=DICE(DICERAW(DICEVECTOR(s)),pos);
pos.key=computeKey(pos);
buildPathMask(pos,path,mask);
}
void SITE(){
int side1,side2;
vector<int> s1,s2;
Position pos;
array<uint64_t,4> path;
array<int,64> mask;
float eval;
vector<moveState> moves;
vector<int> pv;
side1=-1;
while(1){
s1=S(0,3839,0,2399);
while(1){
Sleep(100);
s2=S(0,3839,0,2399);
side2=SIDE(s2[1442+3840*1955]);
if(side2!=-1&&side2!=side1&&WHITE(s2)&&EQUAL(s1,s2))break;
s1=s2;
}
side1=side2;
SET(s2,pos,path,mask);
mctsBatchedMT(pos,path,mask,600,eval,moves,pv,1,side2);
}
}
int main() {
    installCrashDiagnostics();

    try {
        const std::string ptFile = "net.pt";
        const std::string emaFile = "net_ema.pt";
        const std::string planFile = "net.plan";

        std::cout << "Enter FEN (or '960' for a random Chess960 position, '-' for Training):\n";
        std::string fen;
        std::getline(std::cin, fen);

        if (fen == "-") {
            diagLogLine("[main] entering Training()");
            Training(1000000);
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
if(fen=="s")SITE();
        Position pos;
        std::array<uint64_t, 4> path;
        std::array<int, 64> mask;

        if (fen == "960") chess960(pos, path, mask);
        else              fenToPositionPathMask(fen, pos, path, mask);



        MoveList ml;
        int term = 0;
        Position tmp = pos;


        float mctsEvalWhite = 0.5f;
        std::vector<int> pvBeforeRoll;
        std::vector<moveState> rootMoves;
        mctsBatchedMT(pos, path, mask, 10.0, mctsEvalWhite, rootMoves, pvBeforeRoll, 0, 0);

        float v = 0.5f;
        std::vector<float> pol((size_t)POLICY_SIZE, 0.0f);

        g_trt.inferBatch(&pos, 1, &v, pol.data());

        std::cout << "eval=" << v << std::endl;

        for (size_t i = 0; i < pvBeforeRoll.size(); ++i) {
            if (i) std::cout << ' ';
            std::cout << moveToStr(pvBeforeRoll[i]);
        }
        std::cout << "\n";


        std::cout << std::fixed << std::setprecision(6);

        for (const auto& ms : rootMoves) {
            int d = (int)std::to_string(ms.visits).size();
            int spacesBeforePrior = 1 + (to_string(rootMoves[0].visits).size() - d);

            std::cout
                << moveToStr(ms.move)
                << " eval " << ms.eval
                << " visits " << ms.visits
                << std::string(spacesBeforePrior, ' ')
                << "prior " << ms.prior
                << '\n';
        }

        cin.get();

        {
            std::lock_guard<std::mutex> lk(g_trtMutex);
            g_trt.shutdown();
            g_trtReady = false;
        }

        diagLogLine("[main] finished normally");
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
