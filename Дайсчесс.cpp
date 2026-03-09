#include <array>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <iostream>
#include <sstream>
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
#include <thread>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <memory>
#include <deque>
#include <torch/torch.h>
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

using namespace std;



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
        visits.fetch_add(1, std::memory_order_relaxed);
        atomicAddDouble(valueSum, (double)v);
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

    AI_FORCEINLINE bool isExpanded() const {
        return expanded.load(std::memory_order_acquire) != 0;
    }

    AI_FORCEINLINE double sum() const {
        return valueSum.load(std::memory_order_relaxed);
    }
    AI_FORCEINLINE void addVisitAndValue(float v) {
        visits.fetch_add(1, std::memory_order_relaxed);
        atomicAddDouble(valueSum, (double)v);
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
static constexpr int NN_DICE_PLANES = 6;

static constexpr int NN_SQ_PLANES = NN_PIECE_PLANES + NN_EP1_PLANES + NN_EP2_PLANES + NN_CASTLE_PLANES + NN_DICE_PLANES; // 25
static constexpr int NN_INPUT_SIZE = NN_SQ_PLANES * 64; // 1600

using NNInput = array<float, NN_INPUT_SIZE>;

alignas(64) static array<float, 64> NN_PLANE0;
alignas(64) static array<float, 64> NN_PLANE1;
alignas(64) static array<array<float, 64>, 4> NN_DICEPLANE;

static void initNNConstPlanes() {
    NN_PLANE0.fill(0.0f);
    NN_PLANE1.fill(1.0f);
    for (int c = 0; c <= 3; ++c) {
        float v = float(c) * (1.0f / 3.0f);
        NN_DICEPLANE[c].fill(v);
    }
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
        } else {
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
            copyPlane(out, 19 + pt, NN_DICEPLANE[cnt].data());
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
    cout << "side " << pos.side << "\n";
    cout << "dice " << diceIntToFen(pos.dice) << "\n";

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

void makeRandom(Position& pos,TTNode* node) {
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
    pos.dice = Dice[(pos.key+visits)%216];
    if (pawns) {
        if (pos.side == 0)dist = clz64(pawns) >> 3;   // MSVC-safe
        else dist = ctz64(pawns) >> 3;            // MSVC-safe
    }
    for (int i = 0; i < 5; i++)
        while (dicePiece[pos.dice][i] && (pos.color[pos.side] & pos.piece[i]) == 0 && dist > dicePiece[pos.dice][0])
            pos.dice = newDice[pos.dice][i];
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
static constexpr float BN_EPS = 1e-5f;

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



static void cudaCheck(cudaError_t e, const char* expr, const char* file, int line) {
    if (e == cudaSuccess) return;
    std::cerr << "CUDA error: " << cudaGetErrorString(e)
        << " at " << file << ":" << line
        << " in " << expr << "\n";
    std::exit(1);
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
    std::vector<std::vector<float>> blocks;

    nvinfer1::Weights add(std::vector<float>&& v) {
        blocks.push_back(std::move(v));
        auto& b = blocks.back();
        nvinfer1::Weights w{};
        w.type = nvinfer1::DataType::kFLOAT;
        w.values = b.data();
        w.count = (int64_t)b.size();
        return w;
    }

    nvinfer1::Weights addZeros(size_t n) {
        std::vector<float> v(n, 0.0f);
        return add(std::move(v));
    }
};

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

    // IMPORTANT: EXPLICIT_BATCH => s2 is 4D: [N,2C,1,1]
    // Slice along channel axis (dim=1)
    // Your profile is fixed to TRT_MAX_BATCH for MIN/OPT/MAX => we can use TRT_MAX_BATCH here.
    // If you later make batch truly dynamic, you'll need dynamic slice (shape tensors) instead.
    const Dims4 stride{ 1,1,1,1 };

    auto* slW = net.addSlice(*s2,
        Dims4{ 0,   0, 0, 0 },
        Dims4{ TRT_MAX_BATCH, C, 1, 1 },
        stride);
    auto* slB = net.addSlice(*s2,
        Dims4{ 0,   C, 0, 0 },
        Dims4{ TRT_MAX_BATCH, C, 1, 1 },
        stride);
    if (!slW || !slB) return nullptr;

    slW->setName((prefix + ".se.sliceW").c_str());
    slB->setName((prefix + ".se.sliceB").c_str());

    ITensor* Wt = slW->getOutput(0); // [N,C,1,1]
    ITensor* Bt = slB->getOutput(0); // [N,C,1,1]

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

    prof->setDimensions("input", OptProfileSelector::kMIN, Dims4{ TRT_MAX_BATCH, NN_SQ_PLANES, 8, 8 });
    prof->setDimensions("input", OptProfileSelector::kOPT, Dims4{ TRT_MAX_BATCH, NN_SQ_PLANES, 8, 8 });
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

    AI_FORCEINLINE size_t inBytesFull() const {
        return (size_t)maxBatch * (size_t)NN_INPUT_SIZE * sizeof(float);
    }
    AI_FORCEINLINE size_t polBytesFull() const {
        return (size_t)maxBatch * (size_t)POLICY_SIZE * sizeof(float);
    }
    AI_FORCEINLINE size_t valBytesFull() const {
        return (size_t)maxBatch * sizeof(float);
    }

#if AI_HAVE_CUDA_KERNELS
    AI_FORCEINLINE size_t gatherIdxBytesFull() const {
        return (size_t)maxBatch * (size_t)AI_MAX_MOVES * sizeof(int);
    }
    AI_FORCEINLINE size_t gatherLogitsBytesFull() const {
        return (size_t)maxBatch * (size_t)AI_MAX_MOVES * sizeof(float);
    }
#endif

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
        std::memset(hGatherIdxPinned, 0, gatherIdxBytesFull());
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

        // Fixed shape
        if (!ctx->setInputShape("input", nvinfer1::Dims4{ maxBatch, NN_SQ_PLANES, 8, 8 })) {
            std::cerr << "TensorRT: setInputShape(256,25,8,8) failed.\n";
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

        std::cout << "Файл TensorRT плана '" << planFile << "' не найден/не грузится — собираю движок...\n";
        if (!buildAndSavePlan(planFile)) {
            std::cerr << "Не удалось собрать и сохранить TensorRT plan '" << planFile << "'.\n";
            return false;
        }
        std::cout << "Собран и сохранён '" << planFile << "'. Загружаю...\n";

        if (!initFromPlan(planFile)) {
            std::cerr << "Не удалось загрузить TensorRT plan после сборки.\n";
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

    // Fixed-batch execution. Assumes pinned buffers already filled (input, and optional gather idx).
    bool runFixed256AndSync() {
        if (!ctx || !stream) return false;

        if (graphReady && graphExec) {
            CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            return true;
        }

        // Ensure aux streams are set for non-graph enqueue path too
        if (nbAuxStreams > 0 && (int)auxStreams.size() == nbAuxStreams) {
            ctx->setAuxStreams(auxStreams.data(), (int32_t)auxStreams.size());
        }

        CUDA_CHECK(cudaMemcpyAsync(dInput, hInputPinned, inBytesFull(),
            cudaMemcpyHostToDevice, stream));

#if AI_HAVE_CUDA_KERNELS
        CUDA_CHECK(cudaMemcpyAsync(dGatherIdx, hGatherIdxPinned, gatherIdxBytesFull(),
            cudaMemcpyHostToDevice, stream));
#endif

        if (!ctx->enqueueV3(stream)) return false;

#if AI_HAVE_CUDA_KERNELS
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

        CUDA_CHECK(cudaMemcpyAsync(hGatherLogitsPinned, dGatherLogits, gatherLogitsBytesFull(),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(hValuePinned, dValue, valBytesFull(),
            cudaMemcpyDeviceToHost, stream));
#else
        CUDA_CHECK(cudaMemcpyAsync(hPolicyPinned, dPolicy, polBytesFull(),
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(hValuePinned, dValue, valBytesFull(),
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

        // pad
        if (B < maxBatch) {
            const float* src = hInputPinned;
            for (int i = B; i < maxBatch; ++i) {
                std::memcpy(hInputPinned + (size_t)i * (size_t)NN_INPUT_SIZE,
                    src, (size_t)NN_INPUT_SIZE * sizeof(float));
            }
        }

#if AI_HAVE_CUDA_KERNELS
        // If caller didn't provide legal-move indices, just zero them.
        std::memset(hGatherIdxPinned, 0, gatherIdxBytesFull());
#endif

        bool ok = runFixed256AndSync();
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
static constexpr int64_t AI_LOCK_WAIT_US = 2000;   // 2ms (как было)
static constexpr int64_t AI_EXPAND_WAIT_US = 100000;

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
        int lockSpins = 0;

        while (probe < PROBE_LIMIT) {
            MCTSSlot& s = slots[(size_t)idx];

            uint64_t m = s.meta.load(std::memory_order_acquire);
            const uint32_t mg = metaGen(m);
            const uint32_t mt = metaTag(m);

            // Slot from older generation => treat as empty; try to claim it
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

                    s.meta.store(packMeta(g, wantTag), std::memory_order_release);
                    return &n;
                }

                // Someone raced us; re-read same slot
                cpuRelax();
                continue;
            }

            // Same generation
            if (mt == wantTag) {
                if (AI_LIKELY(s.node.key == key)) return &s.node;
                // Rare tag collision -> keep probing
            }

            if (mt == TAG_LOCKED32) {
                using Clock = std::chrono::steady_clock;
                Clock::time_point lockStart = Clock::time_point{};
                uint64_t lockStartIdx = ~0ull;

                // фиксируем "начало ожидания" для конкретного слота idx
                if (lockStartIdx != idx) {
                    lockStartIdx = idx;
                    lockStart = Clock::now();
                    lockSpins = 0;
                }

                // ждём, но ограниченно по времени
                if (Clock::now() - lockStart > std::chrono::microseconds(AI_LOCK_WAIT_US)) {
                    // НИКАКОГО abort: просто сдаёмся на эту попытку (симуляция может повториться)
                    return nullptr;
                }

                backoffWait(lockSpins);
                continue; // IMPORTANT: не двигаем idx и не увеличиваем probe
            }
            lockSpins = 0;

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

                    s.meta.store(packMeta(g, wantTag), std::memory_order_release);
                    return &n;
                }

                // Someone raced us; re-read same slot
                cpuRelax();
                continue;
            }

            // Occupied by other key => go next slot (THIS is what counts as "probe")
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
        int lockSpins = 0;

        while (probe < PROBE_LIMIT) {
            MCTSSlot& s = slots[(size_t)idx];

            const uint64_t m = s.meta.load(std::memory_order_acquire);
            if (AI_UNLIKELY(metaGen(m) != g)) return nullptr;

            const uint32_t mt = metaTag(m);

            if (mt == TAG_LOCKED32) {
                using Clock = std::chrono::steady_clock;
                static thread_local Clock::time_point lockStart = Clock::time_point{};
                static thread_local uint64_t lockStartIdx = ~0ull;

                if (lockStartIdx != idx) {
                    lockStartIdx = idx;
                    lockStart = Clock::now();
                    lockSpins = 0;
                }

                if (Clock::now() - lockStart > std::chrono::microseconds(AI_LOCK_WAIT_US)) {
                    return nullptr;
                }

                backoffWait(lockSpins);
                continue;
            }
            lockSpins = 0;

            if (mt == wantTag) {
                if (AI_LIKELY(s.node.key == key)) return &s.node;
            }
            if (mt == TAG_EMPTY32) return nullptr;

            idx = (idx + 1) & mask;
            ++probe;
        }
        return nullptr;
    }
};

static AI_FORCEINLINE float nodeQ(const TTNode& n) {
    uint32_t v = n.visits.load(std::memory_order_relaxed);
    if (!v) return 0.5f;
    return clamp01((float)(n.sum() / (double)v));
}
static AI_FORCEINLINE float edgeQ(const TTEdge& e) {
    uint32_t v = e.visits.load(std::memory_order_relaxed);
    if (!v) return -1.0f;
    return clamp01((float)(e.sum() / (double)v));
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

// Перевод "value для side-to-move" -> "value для белых"
static AI_FORCEINLINE float toWhitePerspective(float qSideToMove, int sideToMove) {
    // sideToMove: 0=white, 1=black
    return (sideToMove == 0) ? qSideToMove : (1.0f - qSideToMove);
}

static float evalOnePVNoExpandWhite(MCTSTable& T,
    const Position& rootPos,
    const std::array<int, 64>& mask,
    int maxDepth = 256) {
    Position pos = rootPos;

    for (int depth = 0; depth < maxDepth; ++depth) {
        TTNode* n = T.findNodeNoInsert(pos.key);
        if (!n) return 0.5f;

        uint8_t ex = n->expanded.load(std::memory_order_acquire);
        if (ex != 1) {
            // Не ждём, не расширяем — просто используем то, что есть.
            float q = nodeQ(*n);
            return toWhitePerspective(q, pos.side);
        }

        if (n->terminal) {
            // В твоей логике terminal бэкапится как v=1.0 (выигрыш side-to-move).
            float q = 1.0f;
            return toWhitePerspective(q, pos.side);
        }

        if (n->edgeCount == 0) {
            if (n->chance) {
                // Chance-узел: кидаем "кости" как в основном дереве.
                makeRandom(pos,n);
                continue;
            }
            else {
                // Узел без ходов, но не chance: берём средний Q узла.
                float q = nodeQ(*n);
                return toWhitePerspective(q, pos.side);
            }
        }

        // Decision-узел: идём по PV
        TTEdge* e0 = T.edgePtr(n->edgeBegin);
        int bi = selectBestPVEdge(*n, e0);
        int m = e0[bi].move;

        makeMove(pos, mask, m);
    }

    // Если упёрлись в maxDepth — оценим текущий узел как есть
    TTNode* n = T.findNodeNoInsert(pos.key);
    if (!n) return 0.5f;
    float q = nodeQ(*n);
    return toWhitePerspective(q, pos.side);
}

static float evalPVForOneSecondNoExpandWhite(MCTSTable& T,
    const Position& rootPos,
    const std::array<int, 64>& mask,
    double sec = 1.0) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto tEnd = t0 + std::chrono::duration<double>(sec);

    double sum = 0.0;
    uint64_t cnt = 0;

    while (std::chrono::steady_clock::now() < tEnd) {
        float vW = evalOnePVNoExpandWhite(T, rootPos, mask, /*maxDepth*/256);
        sum += (double)vW;
        ++cnt;
    }
    if (!cnt) return 0.5f;
    return (float)(sum / (double)cnt);
}

static AI_FORCEINLINE float cpuctFromVisits(uint32_t parentVisits, bool isRoot) {
    constexpr float C_INIT = 1.25f;
    constexpr float C_BASE = 19652.0f;
    float c = C_INIT + std::log(((float)parentVisits + C_BASE + 1.0f) / C_BASE);
    if (isRoot) c *= 1.10f;
    return c;
}

static AI_FORCEINLINE int selectPUCT(const TTNode& n,
    const TTEdge* e0,
    float cpuct,
    uint32_t parentVisits,
    float parentQ,
    uint32_t rngJitter) {
    constexpr float FPU_REDUCTION = 0.07f;
    const float sqrtN = std::sqrt((float)(parentVisits + 1u));

    float best = -1e30f;
    int bestI = 0;

    const int cnt = (int)n.edgeCount;
    for (int i = 0; i < cnt; ++i) {
        const TTEdge& e = e0[i];
        uint32_t ev = e.visits.load(std::memory_order_relaxed);

        float q;
        if (ev) q = clamp01(e.sum() / (float)ev);
        else    q = clamp01(parentQ - FPU_REDUCTION);

        float u = cpuct * e.prior() * (sqrtN / (1.0f + (float)ev));

        float jit = (float)((rngJitter + (uint32_t)i * 2654435761u) & 1023u)
            * (1.0f / 1023.0f) * 1e-6f;

        float s = q + u + jit;
        if (s > best) { best = s; bestI = i; }
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

    AI_FORCEINLINE void reset() { n = 0; }

    AI_FORCEINLINE TraceStep& push(TTNode* node, TTEdge* edge, bool flip, bool vloss) {
        if (n >= MCTS_MAX_DEPTH) {
            st[MCTS_MAX_DEPTH - 1] = { node, edge, flip, false };
            return st[MCTS_MAX_DEPTH - 1];
        }
        st[n] = { node, edge, flip, vloss };
        return st[n++];
    }
};

struct PendingNN {
    TTNode* leaf = nullptr;
    Position pos;
    MoveList ml;
    Trace trace;
    bool isRoot = false;
};

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

    // Build priors from full policy logits (CHW)
    std::array<float, AI_MAX_MOVES> priors{};
    for (int i = 0; i < cnt; ++i) {
        const int m = p.ml.m[i];
        const int k = policyIndexCHWCanonical(m, p.pos); // 0..4671
        priors[(size_t)i] = polChSq[(size_t)k];
    }

    // Softmax over legal moves
    softmaxLocalArr(priors.data(), cnt);

    // Optional Dirichlet noise at root
    if (p.isRoot) applyRootDirichletNoiseArr(priors.data(), cnt);

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

    // Optional Dirichlet noise at root
    if (p.isRoot) applyRootDirichletNoiseArr(priors.data(), cnt);

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

    // 1) Encode positions into pinned input
    for (int i = 0; i < B; ++i) {
        auto* dst = reinterpret_cast<NNInput*>(
            hInputPinned + (size_t)i * (size_t)NN_INPUT_SIZE
            );
        positionToNNInput(jobs[i].pos, *dst);
    }

    // 2) Pad positions (duplicate slot 0)
    if (B < maxBatch) {
        const float* src = hInputPinned; // slot 0
        for (int i = B; i < maxBatch; ++i) {
            std::memcpy(hInputPinned + (size_t)i * (size_t)NN_INPUT_SIZE,
                src,
                (size_t)NN_INPUT_SIZE * sizeof(float));
        }
    }

#if AI_HAVE_CUDA_KERNELS
    // 3) Build gather indices in move-order for each position
    for (int i = 0; i < B; ++i) {
        const MoveList& ml = jobs[i].ml;
        int* idxBase = hGatherIdxPinned + (size_t)i * (size_t)AI_MAX_MOVES;

        // zero all indices (TRT gather kernel will read up to AI_MAX_MOVES)
        std::memset(idxBase, 0, (size_t)AI_MAX_MOVES * sizeof(int));

        const int n = ml.n;
        for (int j = 0; j < n; ++j) {
            const int m = ml.m[j];
            int k = policyIndexCHWCanonical(m, jobs[i].pos); // CHW: plane*64 + fromSq

            // safety
            if ((unsigned)k >= (unsigned)POLICY_SIZE) k = 0;
            idxBase[j] = k;
        }
    }

    // 4) Pad indices too (duplicate slot 0)
    if (B < maxBatch) {
        const int* src = hGatherIdxPinned; // slot 0
        for (int i = B; i < maxBatch; ++i) {
            std::memcpy(hGatherIdxPinned + (size_t)i * (size_t)AI_MAX_MOVES,
                src,
                (size_t)AI_MAX_MOVES * sizeof(int));
        }
    }
#else
    // No kernels: nothing to do (fallback path uses full policy)
#endif

    // 5) Run fixed-256 execution (CUDA Graph if captured) + sync
    bool ok = runFixed256AndSync();
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
    MCTSTable& T;

    std::atomic<bool> stop{ false };
    std::atomic<int>  qSize{ 0 };

    std::mutex m;
    std::condition_variable cv;

    std::deque<PendingNN> q;   // FIFO
    std::thread th;

    std::vector<float> neutralPol;     // [POLICY_SIZE]
    std::vector<float> neutralLogits;  // [AI_MAX_MOVES]

    explicit InferenceServer(MCTSTable& tab) : T(tab) {
        q.clear();
        // reserve() not available for deque
        neutralPol.assign((size_t)POLICY_SIZE, 0.0f);
        neutralLogits.assign((size_t)AI_MAX_MOVES, 0.0f);
    }

    void start() {
        stop.store(false, std::memory_order_relaxed);
        th = std::thread([this] { this->run(); });
    }

    void stopAndDrain() {
        stop.store(true, std::memory_order_relaxed);
        cv.notify_all();
        if (th.joinable()) th.join();
    }

    int size() const { return qSize.load(std::memory_order_relaxed); }

    void submit(PendingNN&& job) {
        {
            std::lock_guard<std::mutex> lk(m);

            // Pure FIFO:
            q.emplace_back(std::move(job));

            // Optional: mild root priority (still no starvation in practice)
            // if (job.isRoot) q.emplace_front(std::move(job));
            // else            q.emplace_back(std::move(job));

            qSize.fetch_add(1, std::memory_order_relaxed);
        }
        cv.notify_one();
    }

private:
    // pop up to wantB (caller holds lock)
    int popBatchUnlocked(std::vector<PendingNN>& batch, int wantB) {
        batch.clear();
        batch.reserve((size_t)wantB);

        int n = 0;
        while (n < wantB && !q.empty()) {
            batch.emplace_back(std::move(q.front())); // FIFO
            q.pop_front();
            ++n;
        }
        if (n) qSize.fetch_sub(n, std::memory_order_relaxed);
        return n;
    }

    void run() {
        std::vector<PendingNN> batch;
        batch.reserve((size_t)TRT_MAX_BATCH);

        for (;;) {
            // 1) wait for at least 1 job (or stop)
            {
                std::unique_lock<std::mutex> lk(m);
                cv.wait(lk, [&] {
                    return stop.load(std::memory_order_relaxed) || !q.empty();
                    });

                if (stop.load(std::memory_order_relaxed) && q.empty()) break;

                (void)popBatchUnlocked(batch, TRT_MAX_BATCH);
            }

            // 2) small fill window to reach 256 if more jobs arrive
            const auto tFillEnd = std::chrono::steady_clock::now() + std::chrono::microseconds(200);
            while ((int)batch.size() < TRT_MAX_BATCH &&
                std::chrono::steady_clock::now() < tFillEnd) {

                std::unique_lock<std::mutex> lk(m);
                if (q.empty()) {
                    cv.wait_until(lk, tFillEnd, [&] {
                        return stop.load(std::memory_order_relaxed) || !q.empty();
                        });
                }
                if (q.empty()) break;

                std::vector<PendingNN> add;
                const int need = TRT_MAX_BATCH - (int)batch.size();
                (void)popBatchUnlocked(add, need);
                lk.unlock();

                for (auto& j : add) batch.emplace_back(std::move(j));
            }

            const int B = (int)batch.size();
            if (B <= 0) continue;

#if AI_HAVE_CUDA_KERNELS
            bool ok = g_trt.inferBatchGather(batch.data(), B);
            for (int i = 0; i < B; ++i) {
                float v = ok ? g_trt.valueHost(i) : 0.5f;
                const float* logits = ok ? g_trt.gatherLogitsHostPtr(i) : neutralLogits.data();
                expandLeafWithGatheredLogits(T, batch[(size_t)i], v, logits);
            }
#else
            std::vector<Position> posBatch((size_t)B);
            for (int i = 0; i < B; ++i) posBatch[(size_t)i] = batch[(size_t)i].pos;

            bool ok = g_trt.inferBatch(posBatch.data(), B);
            for (int i = 0; i < B; ++i) {
                float v = ok ? g_trt.valueHost(i) : 0.5f;
                const float* pol = ok ? g_trt.policyHostPtr(i) : neutralPol.data();
                expandLeafWithOutputs(T, batch[(size_t)i], v, pol);
            }
#endif
        }

        // drain not needed: loop already drains until q empty after stop=true
    }
};

static void ensureRootExpanded(MCTSTable& T,
    const Position& rootPos,
    const std::array<uint64_t, 4>& path,
    const std::array<int, 64>& mask,
    bool rootNoise,
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

    if (term) {
        root->key = rootPos.key;
        root->edgeBegin = 0;
        root->edgeCount = 0;
        root->terminal = 1;
        root->chance = 0;

        Trace empty; empty.reset();
        backprop(root, 1.0f, empty);
        publishReady(root, rootPos.key, 0, 0, 1, 0);
        return;
    }

    if (ml.n == 0) {
        publishReady(root, rootPos.key, 0, 0, 0, 1);
        return;
    }

    PendingNN p;
    p.leaf = root;
    p.pos = rootPos;
    p.ml = ml;
    p.trace.reset();
    p.isRoot = rootNoise;

    float v = 0.5f;

#if AI_HAVE_CUDA_KERNELS
    if (!g_trt.inferBatchGather(&p, 1)) {
        v = 0.5f;
        std::vector<float> z((size_t)AI_MAX_MOVES, 0.0f);
        expandLeafWithGatheredLogits(T, p, v, z.data());
        return;
    }
    v = g_trt.valueHost(0);
    const float* logits = g_trt.gatherLogitsHostPtr(0);
    expandLeafWithGatheredLogits(T, p, v, logits);
#else
    std::vector<float> pol((size_t)POLICY_SIZE, 0.0f);
    if (!g_trt.inferBatch(&p.pos, 1, &v, pol.data())) {
        v = 0.5f;
        std::fill(pol.begin(), pol.end(), 0.0f);
    }
    expandLeafWithOutputs(T, p, v, pol.data());
#endif
}

static bool runOneSim(MCTSTable& T,
    const Position& rootPos,
    const std::array<uint64_t, 4>& path,
    const std::array<int, 64>& mask,
    bool rootNoise,
    PendingNN& outPending,
    bool& outNeedNN,
    uint32_t rngJitter) {

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
            rollbackVirtualLoss(tr);
            return false;
        }

        TTNode* node = T.getNode(pos.key);
        if (!node) {
            rollbackVirtualLoss(tr);
            return false;
        }

        uint8_t ex = node->expanded.load(std::memory_order_acquire);

        // Someone else expanding
        if (ex == 2) {
            if (!waitWhileExpanding(node)) {
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
                return true;
            }

            if (ml.n == 0) {
                // Chance node
                publishReady(node, pos.key, 0, 0, 0, 1);

                tr.push(node, nullptr, /*flip=*/true, /*vloss=*/false);
                makeRandom(pos,node);
                isRoot = false;
                continue;
            }

            // Need NN
            outNeedNN = true;
            outPending.leaf = node;
            outPending.pos = pos;
            outPending.ml = ml;
            outPending.trace = tr;
            outPending.isRoot = (isRoot && rootNoise);
            return true;
        }

        // Expanded
        if (node->terminal) {
            backprop(node, 1.0f, tr);
            return true;
        }

        if (node->edgeCount == 0) {
            if (node->chance) {
                tr.push(node, nullptr, /*flip=*/true, /*vloss=*/false);
                makeRandom(pos,node);
                isRoot = false;
                continue;
            }
            else {
                float vLeaf = nodeQ(*node);
                backprop(node, vLeaf, tr);
                return true;
            }
        }

        // Decision node: PUCT
        const uint32_t pv = node->visits.load(std::memory_order_relaxed);
        const float parentQ = nodeQ(*node);
        const float cpuct = cpuctFromVisits(pv, isRoot);

        TTEdge* e0 = T.edgePtr(node->edgeBegin);
        int bestI = selectPUCT(*node, e0, cpuct, pv, parentQ, rngJitter);
        TTEdge* e = &e0[bestI];

        // Classic virtual loss (mark the selected edge as "in flight")
        TraceStep& step = tr.push(node, e, /*flip=*/false, /*vloss=*/true);
        applyVirtualLoss(step);

        makeMove(pos, mask, e->move);
        isRoot = false;
    }
}

void mctsBatchedMT(Position& rootPos,
    std::array<uint64_t, 4>& path,
    std::array<int, 64>& mask,
    double timeSec,
    bool rootNoise,
    float& outEvalWhite,
    std::vector<moveState>& outRootMoves) {
    MoveList ml;
    int term;
    genLegal(rootPos, path, mask, ml, term);
    if (term) {
        outEvalWhite = 1 - rootPos.side;
        outRootMoves.push_back({ ml.m[0],outEvalWhite,0 });
        return;
    }

    const size_t nodePow2 = 1ull << 25;
    const size_t edgeCap = 1ull << 27;

    MCTSTable T(nodePow2, edgeCap);

    TTNode* rootNode = T.getNode(rootPos.key);
    if (!rootNode) {
        outEvalWhite = 0.5f;
        outRootMoves.clear();
        return;
    }

    ensureRootExpanded(T, rootPos, path, mask, rootNoise, ml, term);

    // If overflow already happened during root expansion, stop immediately.
    if (T.abort.load(std::memory_order_acquire)) {
        outEvalWhite = 0.5f;
        outRootMoves.clear();
        return;
    }

    InferenceServer nnServer(T);
    nnServer.start();

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const unsigned threads = std::max(1u, hw / 2);

    const auto t0 = std::chrono::steady_clock::now();
    const auto tEnd = t0 + std::chrono::duration<double>(timeSec);

    std::atomic<bool> stop{ false };
    std::atomic<uint64_t> simOK{ 0 }, simFail{ 0 }, nnExp{ 0 };

    auto worker = [&](unsigned tid) {
        uint32_t jitterBase = (uint32_t)(0x9E3779B9u * (tid + 1));
        std::vector<PendingNN> pend;
        pend.reserve((size_t)std::max(1, g_nnBatch));

        uint64_t iter = 0;
        for (;;) {
            if (stop.load(std::memory_order_relaxed)) break;
            if (T.abort.load(std::memory_order_relaxed)) break;

            if (nnServer.size() > 4096) {
                static thread_local int throttleSpins2 = 0;
                throttleOnNNQueue_NoSleep(999999, throttleSpins2); // extreme
                continue;
            }

            if ((iter++ & 63ull) == 0ull) {
                if (std::chrono::steady_clock::now() >= tEnd) break;
            }

            pend.clear();

            const int B = std::max(1, g_nnBatch);
            for (int k = 0; k < B; ++k) {
                if (T.abort.load(std::memory_order_relaxed)) break;

                PendingNN p;
                bool needNN = false;
                static thread_local int throttleSpins = 0;
                int qs = nnServer.size();
                throttleOnNNQueue_NoSleep(qs, throttleSpins);

                // если очередь совсем огромная — не генерим новые leaf'ы прямо сейчас
                if (qs > 2000) continue;
                bool ok = runOneSim(T, rootPos, path, mask, rootNoise,
                    p, needNN,
                    jitterBase + (uint32_t)k * 1337u);
                if (!ok) {
                    simFail.fetch_add(1, std::memory_order_relaxed);
                    if (T.abort.load(std::memory_order_relaxed)) break;
                    continue;
                }

                simOK.fetch_add(1, std::memory_order_relaxed);
                if (needNN) {
                    nnExp.fetch_add(1, std::memory_order_relaxed);
                    nnServer.submit(std::move(p));   // сразу отправили => expanded=2 будет недолго
                }
            }

            if (!pend.empty()) {
                nnExp.fetch_add((uint64_t)pend.size(), std::memory_order_relaxed);
                for (auto& job : pend) nnServer.submit(std::move(job));
            }
            else {
                cpuRelax();
            }
        }
        };

    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (unsigned t = 0; t < threads; ++t) pool.emplace_back(worker, t);

    while (std::chrono::steady_clock::now() < tEnd) {
        if (T.abort.load(std::memory_order_relaxed)) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    stop.store(true, std::memory_order_relaxed);

    for (auto& th : pool) th.join();

    nnServer.stopAndDrain();

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

    (void)simOK; (void)simFail; (void)nnExp;
}

// ===================== TRAINING PATCH BEGIN (FINAL) =====================
// (продолжение будет в сообщении 2/2)
// ===================== TRAINING PATCH BEGIN (FINAL) =====================
// ВСТАВЬ ЭТО ВМЕСТО ТВОЕГО ТЕКУЩЕГО `static void init()` И `int main()`
// (т.е. удалить/заменить всё от `static void init()` до конца файла).



// ========================= Torch BN+SE Net (matches TRT) =========================

// ========================= Torch BN + Affine-SE Net (10x128) =========================
static constexpr double BN_EPS_TORCH = 1e-5;

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
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(C).eps(BN_EPS_TORCH)));

        conv2 = register_module("conv2",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(C, C, 3).padding(1).bias(false)));
        bn2 = register_module("bn2",
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(C).eps(BN_EPS_TORCH)));

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
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(NET_CHANNELS).eps(BN_EPS_TORCH)));

        blocks = register_module("blocks", torch::nn::ModuleList());
        for (int i = 0; i < NET_BLOCKS; ++i) blocks->push_back(ResBlock(NET_CHANNELS));

        polConv1 = register_module("polConv1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(NET_CHANNELS, HEAD_POLICY_C, 1).padding(0).bias(false)));
        polBn1 = register_module("polBn1",
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(HEAD_POLICY_C).eps(BN_EPS_TORCH)));
        polConv2 = register_module("polConv2",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(HEAD_POLICY_C, POLICY_P, 1).padding(0).bias(true)));

        valConv1 = register_module("valConv1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(NET_CHANNELS, HEAD_VALUE_C, 1).padding(0).bias(false)));
        valBn1 = register_module("valBn1",
            torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(HEAD_VALUE_C).eps(BN_EPS_TORCH)));

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
    std::array<float, NN_INPUT_SIZE>  x;      // [1600]

    // sparse policy target:
    // idx = CHW index in [0..POLICY_SIZE-1], i.e. pl*64 + sq
    uint16_t nPi = 0;
    std::array<uint16_t, AI_MAX_MOVES> piIdx{};
    std::array<float, AI_MAX_MOVES> piProb{};
    float q;
    float z = 0.5f; // [0..1] from side-to-move perspective
};

struct ReplayBuffer {
    std::vector<TrainSample> buf;
    size_t cap = 16384;
    size_t head = 0;
    size_t size = 0;

    // Степень "свежести" данных (Prioritized Replay Lite).
    // 1.0  = полностью равномерный выбор (как было).
    // 0.75 = легкий приоритет свежим играм (золотая середина для AlphaZero).
    // 0.5  = сильный перекос в сторону только что сыгранных партий.
    double recent_bias = 0.75;

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

    bool sampleBatch(std::vector<TrainSample>& out, int B, std::mt19937& rng) {
        // 1. Изменяем размер вектора ДО захвата мьютекса.
        // Это убирает микро-фризы (Lock Contention), позволяя Self-play потокам
        // быстрее складывать новые партии в буфер.
        out.resize((size_t)B);

        std::lock_guard<std::mutex> lk(m);
        if (size < (size_t)B) return false;

        // Используем непрерывное распределение [0.0, 1.0)
        std::uniform_real_distribution<double> d(0.0, 1.0);

        auto phys = [&](size_t logical) -> size_t {
            size_t start = (head + cap - size) % cap;
            return (start + logical) % cap;
            };

        for (int i = 0; i < B; ++i) {
            double u = d(rng);

            // 2. Математический трюк: возводим 'u' в степень < 1.0.
            // График функции y = x^0.75 выгибается вверх. 
            // Это значит, что случайные значения будут чаще смещаться ближе к 1.0
            // Логический индекс 0 — самая старая позиция, (size - 1) — самая новая.
            size_t li = (size_t)(size * std::pow(u, recent_bias));

            if (li >= size) li = size - 1; // Защита от выхода за пределы

            out[(size_t)i] = buf[phys(li)];
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

    if (!newCtx->setInputShape("input", Dims4{ trt.maxBatch, NN_SQ_PLANES, 8, 8 })) {
        std::cerr << "TensorRT: setInputShape failed on new ctx.\n";
        delete newCtx;
        return false;
    }

    if (trt.ctx) { delete trt.ctx; trt.ctx = nullptr; }
    trt.ctx = newCtx;

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
            float s = g[i] / std::sqrt(v[i] + BN_EPS);
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

struct InferInFlightGuard {
    InferInFlightGuard() { g_inferInFlight.fetch_add(1, std::memory_order_relaxed); }
    ~InferInFlightGuard() { g_inferInFlight.fetch_sub(1, std::memory_order_relaxed); }
};
struct InferenceServerTrain {
    MCTSTable& T;

    std::atomic<bool> stop{ false };
    std::atomic<int>  qSize{ 0 };
    std::atomic<int>  busy{ 0 };

    std::mutex m;
    std::condition_variable cv;

    std::deque<PendingNN> q;   // FIFO
    std::thread th;

    std::vector<float> neutralPol;
    std::vector<float> neutralLogits;

    explicit InferenceServerTrain(MCTSTable& tab) : T(tab) {
        q.clear();
        neutralPol.assign((size_t)POLICY_SIZE, 0.0f);
        neutralLogits.assign((size_t)AI_MAX_MOVES, 0.0f);
    }

    void start() {
        stop.store(false, std::memory_order_relaxed);
        th = std::thread([this] { this->run(); });
    }

    void requestStop() {
        stop.store(true, std::memory_order_relaxed);
        cv.notify_all();
    }

    void join() {
        if (th.joinable()) th.join();
    }

    int size() const { return qSize.load(std::memory_order_relaxed); }

    void submit(PendingNN&& job) {
        {
            std::lock_guard<std::mutex> lk(m);

            // Pure FIFO:
            q.emplace_back(std::move(job));

            // Optional: mild root priority
            // if (job.isRoot) q.emplace_front(std::move(job));
            // else            q.emplace_back(std::move(job));

            qSize.fetch_add(1, std::memory_order_relaxed);
        }
        cv.notify_one();
    }

    void waitIdle() {
        for (;;) {
            if (size() == 0 && busy.load(std::memory_order_relaxed) == 0) break;
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }

    // Use only when server is idle.
    void clearQueueUnsafeWhenIdle() {
        std::lock_guard<std::mutex> lk(m);
        q.clear();
        qSize.store(0, std::memory_order_relaxed);
    }

private:
    bool popBatchUnlocked(std::vector<PendingNN>& batch, int wantB) {
        batch.clear();
        batch.reserve((size_t)wantB);

        int n = 0;
        while (n < wantB && !q.empty()) {
            batch.emplace_back(std::move(q.front())); // FIFO
            q.pop_front();
            ++n;
        }
        if (n) qSize.fetch_sub(n, std::memory_order_relaxed);
        return n != 0;
    }

    void run() {
        std::vector<PendingNN> batch;
        batch.reserve((size_t)TRT_MAX_BATCH);

        for (;;) {
            // wait for work or stop
            {
                std::unique_lock<std::mutex> lk(m);
                busy.store(0, std::memory_order_relaxed);

                cv.wait_for(lk, std::chrono::milliseconds(1), [&] {
                    return stop.load(std::memory_order_relaxed) || !q.empty();
                    });

                if (stop.load(std::memory_order_relaxed) && q.empty()) break;
                if (q.empty()) continue;

                busy.store(1, std::memory_order_relaxed);
                (void)popBatchUnlocked(batch, TRT_MAX_BATCH);
            }

            const int B = (int)batch.size();
            if (B <= 0) continue;

#if AI_HAVE_CUDA_KERNELS
            bool ok = false;
            {
                InferInFlightGuard ig;
                std::lock_guard<std::mutex> lk(g_trtMutex);
                ok = g_trt.inferBatchGather(batch.data(), B);
            }

            for (int i = 0; i < B; ++i) {
                float v = ok ? g_trt.valueHost(i) : 0.5f;
                const float* logits = ok ? g_trt.gatherLogitsHostPtr(i) : neutralLogits.data();
                expandLeafWithGatheredLogits(T, batch[(size_t)i], v, logits);
            }
#else
            std::vector<Position> posBatch((size_t)B);
            for (int i = 0; i < B; ++i) posBatch[(size_t)i] = batch[(size_t)i].pos;

            bool ok = false;
            {
                InferInFlightGuard ig;
                std::lock_guard<std::mutex> lk(g_trtMutex);
                ok = g_trt.inferBatch(posBatch.data(), B);
            }

            for (int i = 0; i < B; ++i) {
                float v = ok ? g_trt.valueHost(i) : 0.5f;
                const float* pol = ok ? g_trt.policyHostPtr(i) : neutralPol.data();
                expandLeafWithOutputs(T, batch[(size_t)i], v, pol);
            }
#endif
        }

        // Drain remaining (best effort)
        for (;;) {
            std::vector<PendingNN> tail;
            {
                std::lock_guard<std::mutex> lk(m);
                if (q.empty()) break;
                busy.store(1, std::memory_order_relaxed);
                (void)popBatchUnlocked(tail, TRT_MAX_BATCH);
            }

            const int B = (int)tail.size();
            if (B <= 0) break;

#if AI_HAVE_CUDA_KERNELS
            bool ok = false;
            {
                InferInFlightGuard ig;
                std::lock_guard<std::mutex> lk(g_trtMutex);
                ok = g_trt.inferBatchGather(tail.data(), B);
            }

            for (int i = 0; i < B; ++i) {
                float v = ok ? g_trt.valueHost(i) : 0.5f;
                const float* logits = ok ? g_trt.gatherLogitsHostPtr(i) : neutralLogits.data();
                expandLeafWithGatheredLogits(T, tail[(size_t)i], v, logits);
            }
#else
            std::vector<Position> posBatch((size_t)B);
            for (int i = 0; i < B; ++i) posBatch[(size_t)i] = tail[(size_t)i].pos;

            bool ok = false;
            {
                InferInFlightGuard ig;
                std::lock_guard<std::mutex> lk(g_trtMutex);
                ok = g_trt.inferBatch(posBatch.data(), B);
            }

            for (int i = 0; i < B; ++i) {
                float v = ok ? g_trt.valueHost(i) : 0.5f;
                const float* pol = ok ? g_trt.policyHostPtr(i) : neutralPol.data();
                expandLeafWithOutputs(T, tail[(size_t)i], v, pol);
            }
#endif
        }

        busy.store(0, std::memory_order_relaxed);
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
        // cur обновится compare_exchange_weak'ом
    }
    return false;
}

static AI_FORCEINLINE void refundSimBudget(std::atomic<int>& simsLeft) {
    simsLeft.fetch_add(1, std::memory_order_relaxed);
}
struct SearchPool {
    std::vector<std::thread> pool;
    std::mutex m;
    std::condition_variable cv;
    bool stop = false;
    std::atomic<bool> cancelJob{ false };
    // job dispatch
    int jobId = 0;
    std::atomic<int> workersBusy{ 0 };
    std::atomic<int> simsLeft{ 0 };

    // job params (valid only during active job)
    MCTSTable* T = nullptr;
    InferenceServerTrain* srv = nullptr;
    const Position* rootPos = nullptr;
    const std::array<uint64_t, 4>* path = nullptr;
    const std::array<int, 64>* mask = nullptr;

    unsigned threads = 1;

    void start(unsigned nThreads) {
        stop = false;
        threads = std::max(1u, nThreads);
        pool.reserve(threads);

        for (unsigned tid = 0; tid < threads; ++tid) {
            pool.emplace_back([this, tid] { this->workerMain(tid); });
        }
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lk(m);
            stop = true;
        }
        cv.notify_all();
        for (auto& th : pool) if (th.joinable()) th.join();
        pool.clear();
    }

    void runSims(MCTSTable& TT,
        InferenceServerTrain& server,
        const Position& rp,
        const std::array<uint64_t, 4>& pth,
        const std::array<int, 64>& msk,
        int sims) {
        if (pool.empty()) return;

        if (TT.abort.load(std::memory_order_relaxed)) return;

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
            ++jobId;
        }
        cv.notify_all();

        // ВАЖНО: даже если TT.abort == true, мы всё равно ждём,
        // чтобы воркеры гарантированно вышли, иначе нельзя делать T.newGame().
        const auto t0 = std::chrono::steady_clock::now();
        const auto hardTimeout = std::chrono::seconds(2);

        for (;;) {
            if (workersBusy.load(std::memory_order_relaxed) == 0) break;

            if (TT.abort.load(std::memory_order_relaxed)) {
                // ускоряем останов
                cancelJob.store(true, std::memory_order_relaxed);
                simsLeft.store(0, std::memory_order_relaxed);
            }

            if (std::chrono::steady_clock::now() - t0 > hardTimeout) {
                // Если реально зависли — лучше остановить пул, чем продолжать с битым состоянием.
                std::cerr << "[SearchPool] ERROR: workers did not stop in time. Forcing shutdown.\n";
                shutdown();
                break;
            }

            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }

private:
    void workerMain(unsigned tid) {
        int myJob = 0;
        uint32_t jitterBase = (uint32_t)(0x9E3779B9u * (tid + 1));

        for (;;) {
            MCTSTable* TT = nullptr;
            InferenceServerTrain* server = nullptr;
            const Position* rp = nullptr;
            const std::array<uint64_t, 4>* pth = nullptr;
            const std::array<int, 64>* msk = nullptr;

            {
                std::unique_lock<std::mutex> lk(m);
                cv.wait(lk, [&] { return stop || jobId != myJob; });
                if (stop) return;

                myJob = jobId;
                TT = T;
                server = srv;
                rp = rootPos;
                pth = path;
                msk = mask;
            }

            if (!TT || TT->abort.load(std::memory_order_relaxed)) {
                workersBusy.fetch_sub(1, std::memory_order_relaxed);
                continue;
            }

            // execute sims
int k = 0;
for (;;) {
    if (TT->abort.load(std::memory_order_relaxed)) break;
    if (cancelJob.load(std::memory_order_relaxed)) break;

    // avoid unbounded queue growth
    static thread_local int throttleSpins = 0;
    int qs = server ? server->size() : 0;
    throttleOnNNQueue_NoSleep(qs, throttleSpins);

    // IMPORTANT:
    // do NOT spend sim budget while we are only throttling on NN queue pressure
    if (qs > 2000) {
        cpuRelax();
        continue;
    }

    // claim exactly one simulation budget item
    if (!tryClaimSimBudget(simsLeft)) break;

    PendingNN p;
    bool needNN = false;

    bool ok = runOneSim(*TT, *rp, *pth, *msk, /*rootNoise=*/false,
        p, needNN,
        jitterBase + (uint32_t)(k++) * 1337u);

    if (!ok) {
        // simulation did not actually happen -> give budget back
        refundSimBudget(simsLeft);

        if (TT->abort.load(std::memory_order_relaxed)) break;
        if (cancelJob.load(std::memory_order_relaxed)) break;

        cpuRelax();
        continue;
    }

    if (needNN && server) server->submit(std::move(p));
}

            workersBusy.fetch_sub(1, std::memory_order_relaxed);
        }
    }
};

// ------------------------------------------------------------
// Search fixed number of simulations (sims) with tree reuse
// Dirichlet noise применяется ТОЛЬКО временно на root (не портит priors в TT навсегда)
// ------------------------------------------------------------

// Expand root (or any node keyed by rootPos) exactly once for training-selfplay.
// IMPORTANT:
//  - does NOT apply Dirichlet noise (p.isRoot = false)
//  - marks GPU inference as "in flight" so trainer yields (InferInFlightGuard)
//  - protects TensorRT with g_trtMutex
static void ensureExpandedTrain(MCTSTable& T,
    const Position& rootPos,
    const std::array<uint64_t, 4>& path,
    const std::array<int, 64>& mask) {
    if (T.abort.load(std::memory_order_relaxed)) return;

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

    // Generate legal moves for this position (note: genLegal mutates Position)
    MoveList ml;
    int term = 0;
    Position tmp = rootPos;

    genLegal(tmp,
        path,
        mask,
        ml, term);

    if (term) {
        root->key = rootPos.key;
        root->edgeBegin = 0;
        root->edgeCount = 0;
        root->terminal = 1;
        root->chance = 0;

        Trace empty; empty.reset();
        backprop(root, 1.0f, empty);
        publishReady(root, rootPos.key, 0, 0, 1, 0);
        return;
    }

    if (ml.n == 0) {
        // Chance node (dice roll)
        publishReady(root, rootPos.key, 0, 0, 0, 1);
        return;
    }

    PendingNN p;
    p.leaf = root;
    p.pos = rootPos;
    p.ml = ml;
    p.trace.reset();
    p.isRoot = false; // IMPORTANT: no Dirichlet noise in permanent expansion

    float v = 0.5f;

#if AI_HAVE_CUDA_KERNELS
    // Gathered-logits path
    {
        bool ok = false;
        {
            InferInFlightGuard ig;             // trainer yields while TRT is running
            std::lock_guard<std::mutex> lk(g_trtMutex);
            ok = g_trt.inferBatchGather(&p, 1);
        }

        v = ok ? g_trt.valueHost(0) : 0.5f;

        if (ok) {
            const float* logits = g_trt.gatherLogitsHostPtr(0);
            expandLeafWithGatheredLogits(T, p, v, logits);
            return;
        }
    }

    // If TRT failed: expand with neutral logits (all zeros)
    {
        std::vector<float> z((size_t)AI_MAX_MOVES, 0.0f);
        expandLeafWithGatheredLogits(T, p, v, z.data());
    }
#else
    // Full-policy path (no custom kernels)
    std::vector<float> pol((size_t)POLICY_SIZE, 0.0f);
    bool ok = false;
    {
        InferInFlightGuard ig;
        std::lock_guard<std::mutex> lk(g_trtMutex);
        ok = g_trt.inferBatch(&p.pos, 1, &v, pol.data());
    }
    if (!ok) {
        v = 0.5f;
        std::fill(pol.begin(), pol.end(), 0.0f);
    }
    expandLeafWithOutputs(T, p, v, pol.data());
#endif
}

static void collectRootMoves(MCTSTable& T,
    const Position& rootPos,
    float& outQSideToMove,
    std::vector<moveState>& outMoves) {
    TTNode* root = T.getNode(rootPos.key);
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
        if (v) ev = clamp01(e.sum() / (float)v);

        outMoves.push_back(moveState{ e.move, ev, (int)v });
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

    double sum = 0.0;
    std::vector<double> w(mv.size());
    for (size_t i = 0; i < mv.size(); ++i) {
        double v = (double)std::max(0, mv[i].visits);
        double ww = std::pow(v + 1e-9, 1.0 / (double)temperature);
        w[i] = ww;
        sum += ww;
    }
    if (!(sum > 0.0)) return mv[0].move;

    std::uniform_real_distribution<double> d(0.0, sum);
    double r = d(Random);

    double acc = 0.0;
    for (size_t i = 0; i < mv.size(); ++i) {
        acc += w[i];
        if (r <= acc) return mv[i].move;
    }
    return mv.back().move;
}

// policy target — SPARSE (idx/prob), idx в CHW: k=pl*64+sq
static void buildSparsePolicyTargetCHW(const Position& pos,
    const std::vector<moveState>& mv,
    uint16_t& outN,
    std::array<uint16_t, AI_MAX_MOVES>& outIdx,
    std::array<float, AI_MAX_MOVES>& outProb) {
    outN = 0;
    outIdx.fill(0);
    outProb.fill(0.0f);

    if (mv.empty()) return;

    const int n = std::min((int)mv.size(), AI_MAX_MOVES);

    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += (double)std::max(0, mv[(size_t)i].visits);

    if (!(sum > 0.0)) {
        float inv = 1.0f / (float)n;
        outN = (uint16_t)n;
        for (int i = 0; i < n; ++i) {
            int k = policyIndexCHWCanonical(mv[(size_t)i].move, pos);
            outIdx[(size_t)i] = (uint16_t)k;            outProb[(size_t)i] = inv;
        }
        return;
    }

    float inv = (float)(1.0 / sum);
    outN = (uint16_t)n;
    for (int i = 0; i < n; ++i) {
        int k = policyIndexCHWCanonical(mv[(size_t)i].move, pos);
        outIdx[(size_t)i] = (uint16_t)k;        outProb[(size_t)i] = (float)std::max(0, mv[(size_t)i].visits) * inv;
    }
}

// временно (на один search) зашумливаем root priors и потом откатываем назад
static void runFixedSims(MCTSTable& T,
    SearchPool& pool,
    InferenceServerTrain& srv,
    const Position& rootPos,
    const std::array<uint64_t, 4>& path,
    const std::array<int, 64>& mask,
    int sims,
    bool rootNoise) {
    if (T.abort.load(std::memory_order_relaxed)) return;

    ensureExpandedTrain(T, rootPos, path, mask);
    if (T.abort.load(std::memory_order_relaxed)) return;

    TTNode* root = T.getNode(rootPos.key);
    TTEdge* e0 = nullptr;
    int nEdges = 0;

    // Сохраняем root priors в сыром квантованном виде, чтобы потом
    // восстановить их без лишней ошибки округления.
    std::vector<uint16_t> savedPriorQ;

    if (rootNoise &&
        root &&
        root->expanded.load(std::memory_order_acquire) == 1 &&
        root->edgeCount >= 2) {
        e0 = T.edgePtr(root->edgeBegin);
        nEdges = (int)root->edgeCount;

        savedPriorQ.resize((size_t)nEdges);
        std::vector<float> noisy((size_t)nEdges);

        for (int i = 0; i < nEdges; ++i) {
            savedPriorQ[(size_t)i] = e0[i].priorRaw();
            noisy[(size_t)i] = e0[i].prior();
        }

        applyRootDirichletNoise(noisy);

        for (int i = 0; i < nEdges; ++i) {
            e0[i].setPrior(noisy[(size_t)i]);
        }
    }

    pool.runSims(T, srv, rootPos, path, mask, sims);

    srv.waitIdle();

    // Восстанавливаем исходные root priors.
    if (!savedPriorQ.empty() && e0 && nEdges > 0) {
        for (int i = 0; i < nEdges; ++i) {
            e0[i].setPriorRaw(savedPriorQ[(size_t)i]);
        }
    }
}

// ------------------------------------------------------------
// Self-play: переиспользуем один MCTSTable + один InferenceServerTrain + SearchPool
// ------------------------------------------------------------

static AI_FORCEINLINE void resetMCTSTableForNewGame(MCTSTable& T) {
    // O(1) reset via generation counter
    T.newGame();
}

struct SelfPlayContext {
    MCTSTable T;
    InferenceServerTrain server;
    SearchPool pool;

    explicit SelfPlayContext(size_t nodePow2, size_t edgeCap)
        : T(nodePow2, edgeCap), server(T) {
    }

    void start() {
        server.start();

        // Heuristic: limit threads to avoid excessive contention (tune if you want).
        unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        unsigned n = std::min(hw, 8u);
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
};

static void selfPlayOneGame960(SelfPlayContext& sp,
    ReplayBuffer& rb,
    int simsPerPos,
    int maxPlies,
    bool addRootNoise,
    int& outPlyCount,
    bool& outTerminated) {
    sp.resetForNewGame();

    Position pos;
    array<uint64_t, 4> path;
    array<int, 64> mask;

    std::vector<TrainSample> game;
    std::vector<int> sideAtSample;

    MoveList ml;
    int term = 0;

    std::vector<moveState> moves;

    TrainSample sample;
    int d = 0;

    chess960(pos, path, mask);

    game.reserve((size_t)maxPlies);
    sideAtSample.reserve((size_t)maxPlies);

    outTerminated = false;

    for (int ply = 0; ply < maxPlies; ++ply) {
        // Early stop if table overflow
        if (sp.T.abort.load(std::memory_order_relaxed)) break;

        genLegal(pos, path, mask, ml, term);

        if (term) { outTerminated = true; break; }

        if (ml.n == 0) {
            makeRandom(pos,sp.T.findNodeNoInsert(pos.key));
            continue;
        }

        bool rootNoiseHere = addRootNoise && (d < 20);

        runFixedSims(sp.T, sp.pool, sp.server, pos, path, mask, simsPerPos, rootNoiseHere);
        if (sp.T.abort.load(std::memory_order_relaxed)) break;

        collectRootMoves(sp.T, pos, sample.q, moves);

        if (moves.empty()) break;

        positionToNNInput(pos, sample.x);
        buildSparsePolicyTargetCHW(pos, moves, sample.nPi, sample.piIdx, sample.piProb);

        game.push_back(sample);
        sideAtSample.push_back(pos.side);

        float temp = (d < 20) ? 1.0f : 0.0f;
        int mv = pickMoveFromVisits(moves, temp);
        if (!mv) break;

        makeMove(pos, mask, mv);
        ++d;
    }

    outPlyCount = d;

    float zWhite = 0.5f;
    if (outTerminated) {
        // term означает "у side-to-move есть немедленная победа (взятие короля)".
        // winner = side-to-move => whiteWin = 1 - pos.side
        zWhite = 1.0f - pos.side;
    }
    else return;

    for (size_t i = 0; i < game.size(); ++i) {
        int stm = sideAtSample[i];
        float zi = (stm == 0) ? zWhite : (1.0f - zWhite);
        game[i].z = 0.5f * zi + 0.5f * game[i].q;
        rb.push(game[i]);
    }
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

struct Trainer {
    torch::Device device{ torch::kCPU };
    bool useCuda = false;

    // Hyperparams
    double initial_lr = 2e-4;
    double current_lr = initial_lr;
    double wd = 1e-4;
// restart-warmup
uint64_t resumeStartStep = 0;              // step at process start / resume
uint64_t warmupStepsAfterRestart = 2000;   // tune: 1000..5000
double   warmupStartFactor = 0.10;         // 0.05..0.25 usually good
    // LR schedule
    std::vector<uint64_t> lr_milestones = { 300000, 600000, 850000 };
    double lr_gamma = 0.5;

    // Batch
    int B = 256;

    // RNG
    std::mt19937 rng{ 0xBADC0FFEu };

    // Optimizer
    std::unique_ptr<torch::optim::AdamW> opt;

    // CPU staging (pinned if CUDA)
    torch::Tensor xCPU, idxCPU, probCPU, zCPU;

    // Device tensors (allocated once)
    torch::Tensor xDev, idxDev, probDev, zDev;

    // State
    uint64_t steps = 0;
    float lastLoss = 0.0f;
    float lastLossP = 0.0f; // <--- ДОБАВИТЬ ЭТО
    float lastLossV = 0.0f; // <--- ДОБАВИТЬ ЭТО
double computeBaseLRFromSteps(uint64_t s) const {
    double lr = initial_lr;
    for (uint64_t ms : lr_milestones) {
        if (s >= ms) lr *= lr_gamma;
    }
    return lr;
}
double computeRestartWarmupMultiplier(uint64_t s) const {
    if (warmupStepsAfterRestart == 0) return 1.0;

    uint64_t delta = (s >= resumeStartStep) ? (s - resumeStartStep) : 0;
    if (delta >= warmupStepsAfterRestart) return 1.0;

    double t = (double)delta / (double)warmupStepsAfterRestart; // 0..1
    return warmupStartFactor + (1.0 - warmupStartFactor) * t;
}
    void updateLR(bool forceLog = false) {
    double base_lr = computeBaseLRFromSteps(steps);
    double warm_mult = computeRestartWarmupMultiplier(steps);
    double target_lr = base_lr * warm_mult;

    if (forceLog || std::fabs(target_lr - current_lr) > 1e-15) {
        current_lr = target_lr;
        for (auto& group : opt->param_groups()) {
            static_cast<torch::optim::AdamWOptions&>(group.options()).lr(current_lr);
        }

        std::cerr << "[Trainer] LR=" << current_lr
                  << " (base=" << base_lr
                  << ", warmup_x=" << warm_mult
                  << ", step=" << steps << ")\n";
    }
}

    void init(Net& model) {
        // Stop libtorch from stealing CPU threads
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

        {
            std::lock_guard<std::mutex> lk(g_modelMutex);
            model->to(device);
            model->train();
        }

        opt = std::make_unique<torch::optim::AdamW>(
            model->parameters(),
            torch::optim::AdamWOptions(initial_lr).weight_decay(wd)
        );
resumeStartStep = steps;   // important for restart-warmup
current_lr = -1.0;        // force apply
updateLR(true);
        auto makeCPU = [&](torch::IntArrayRef sizes, torch::ScalarType t) {
            auto ten = torch::empty(sizes, torch::TensorOptions().dtype(t).device(torch::kCPU));
            if (useCuda) ten = ten.pin_memory();
            return ten;
            };

        xCPU = makeCPU({ B, NN_SQ_PLANES, 8, 8 }, torch::kFloat32);
        idxCPU = makeCPU({ B, AI_MAX_MOVES }, torch::kInt64);
        probCPU = makeCPU({ B, AI_MAX_MOVES }, torch::kFloat32);
        zCPU = makeCPU({ B, 1 }, torch::kFloat32);

        if (useCuda) {
            xDev = torch::empty({ B, NN_SQ_PLANES, 8, 8 }, torch::TensorOptions().dtype(torch::kFloat32).device(device));
            idxDev = torch::empty({ B, AI_MAX_MOVES }, torch::TensorOptions().dtype(torch::kInt64).device(device));
            probDev = torch::empty({ B, AI_MAX_MOVES }, torch::TensorOptions().dtype(torch::kFloat32).device(device));
            zDev = torch::empty({ B, 1 }, torch::TensorOptions().dtype(torch::kFloat32).device(device));
        }
        else {
            // CPU mode: alias
            xDev = xCPU; idxDev = idxCPU; probDev = probCPU; zDev = zCPU;
        }
    }

    // Train block with a strict time budget.
    int trainBlockBudgetMs(ReplayBuffer& rb, Net& model,
        int budgetMs,
        int maxStepsHard,
    int warmupBatches = 1000) {
        if (budgetMs <= 0 || maxStepsHard <= 0) return 0;

        const size_t need = (size_t)B * (size_t)std::max(1, warmupBatches);
        if (rb.currentSize() < need) return 0;

        const auto tEnd = std::chrono::steady_clock::now() + std::chrono::milliseconds(budgetMs);

        std::vector<TrainSample> batch;
        batch.reserve((size_t)B);

        int done = 0;

        // Distributions outside inner loops (micro-optimization + cleaner)
        std::bernoulli_distribution coin(0.5);

        for (int it = 0; it < maxStepsHard; ++it) {
            if (std::chrono::steady_clock::now() >= tEnd) break;
            if (!rb.sampleBatch(batch, B, rng)) break;

            // ---- Fill CPU staging ----
            float* xp = xCPU.data_ptr<float>();
            int64_t* ip = idxCPU.data_ptr<int64_t>();
            float* pp = probCPU.data_ptr<float>();
            float* zp = zCPU.data_ptr<float>();

            for (int i = 0; i < B; ++i) {
                const TrainSample& s = batch[(size_t)i];

                std::memcpy(xp + (size_t)i * (size_t)NN_INPUT_SIZE,
                    s.x.data(),
                    (size_t)NN_INPUT_SIZE * sizeof(float));

                // sparse policy target packed into fixed 255 slots (rest are already 0 in samples)
                for (int j = 0; j < AI_MAX_MOVES; ++j) {
                    ip[(size_t)i * (size_t)AI_MAX_MOVES + (size_t)j] = (int64_t)s.piIdx[(size_t)j];
                    pp[(size_t)i * (size_t)AI_MAX_MOVES + (size_t)j] = s.piProb[(size_t)j];
                }

                // zCPU is [B,1] contiguous, so zp[i] is fine
                zp[(size_t)i] = s.z;
            }


            

            // ---- H2D (no realloc) ----
            if (useCuda) {
                xDev.copy_(xCPU, /*non_blocking=*/true);
                idxDev.copy_(idxCPU, /*non_blocking=*/true);
                probDev.copy_(probCPU, /*non_blocking=*/true);
                zDev.copy_(zCPU, /*non_blocking=*/true);
            }

            float lossScalar = 0.0f;
            float lossPScalar = 0.0f; // <--- ДОБАВИТЬ
            float lossVScalar = 0.0f; // <--- ДОБАВИТЬ
            bool didStep = false;

            {
                std::lock_guard<std::mutex> lk(g_modelMutex);

                auto out = model->forward(xDev);
                auto pol = out.first;       // [B,73,8,8]
                auto valLogits = out.second; // [B,1] logits in train()

                auto polFlat = pol.flatten(1);                 // [B,4672]
                auto logp = torch::log_softmax(polFlat, 1); // [B,4672]
                auto g = logp.gather(1, idxDev);         // [B,255]

                auto lossP = -(probDev * g).sum(1).mean();
                auto lossV = torch::binary_cross_entropy_with_logits(valLogits, zDev);
                auto loss = lossP + lossV;

                if (torch::isfinite(loss).item<bool>()) {
                    opt->zero_grad();
                    loss.backward();
                    torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
                    opt->step();

                    lossScalar = loss.item<float>();
                    lossPScalar = lossP.item<float>(); // <--- СОХРАНЯЕМ lossP
                    lossVScalar = lossV.item<float>(); // <--- СОХРАНЯЕМ lossV
                    didStep = true;
                }
            }

            if (!didStep) break;

            ++done;
            ++steps;
            lastLoss = lossScalar;
            lastLossP = lossPScalar; // <--- ОБНОВЛЯЕМ STATE ТРЕНЕРА
            lastLossV = lossVScalar; // <--- ОБНОВЛЯЕМ STATE ТРЕНЕРА
            updateLR();
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
// Trainer checkpoint: optimizer state + trainer.steps
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

struct TrainerStateDisk {
    uint64_t magic = 0x545241494E535445ULL; // arbitrary magic
    uint32_t version = 1;
    uint32_t reserved = 0;
    uint64_t steps = 0;
};

static bool saveTrainerState(const std::string& stateFile, const Trainer& trainer) {
    TrainerStateDisk s;
    s.steps = trainer.steps;
    return writeFileAll(stateFile, &s, sizeof(s));
}

static bool loadTrainerState(const std::string& stateFile, Trainer& trainer) {
    std::vector<char> blob;
    readFileAll(stateFile, blob);
    if (blob.size() != sizeof(TrainerStateDisk)) return false;

    TrainerStateDisk s{};
    std::memcpy(&s, blob.data(), sizeof(s));

    if (s.magic != 0x545241494E535445ULL) return false;
    if (s.version != 1) return false;

    trainer.steps = s.steps;
    return true;
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
    else {
        try {
            torch::save(model, ptFile);
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "torch::save (create) failed: " << e.what() << "\n";
            return false;
        }
    }
}
static inline bool isFiniteF(float x) {
    return std::isfinite((double)x) != 0;
}

static void softmaxTo(std::vector<float>& out, const float* logits, int n) {
    out.resize((size_t)n);
    if (n <= 0) return;

    float mx = logits[0];
    for (int i = 1; i < n; ++i) mx = std::max(mx, logits[i]);

    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        double e = std::exp((double)logits[i] - (double)mx);
        out[(size_t)i] = (float)e;
        sum += e;
    }

    if (!(sum > 0.0)) {
        float inv = 1.0f / (float)n;
        for (int i = 0; i < n; ++i) out[(size_t)i] = inv;
        return;
    }

    float inv = (float)(1.0 / sum);
    for (int i = 0; i < n; ++i) out[(size_t)i] *= inv;
}

static double klDiv(const std::vector<float>& p, const std::vector<float>& q, double eps = 1e-12) {
    // KL(p||q) with epsilon floor
    const int n = (int)p.size();
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        double pi = std::max((double)p[(size_t)i], eps);
        double qi = std::max((double)q[(size_t)i], eps);
        s += pi * (std::log(pi) - std::log(qi));
    }
    return s;
}

static double l1Dist(const std::vector<float>& p, const std::vector<float>& q) {
    const int n = (int)p.size();
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += std::fabs((double)p[(size_t)i] - (double)q[(size_t)i]);
    return s;
}

static void topKIndices(const float* x, int n, int K, std::vector<int>& outIdx) {
    outIdx.clear();
    outIdx.reserve((size_t)K);


    std::vector<int> idx((size_t)n);
    for (int i = 0; i < n; ++i) idx[(size_t)i] = i;

    std::partial_sort(idx.begin(), idx.begin() + std::min(K, n), idx.end(),
        [&](int a, int b) { return x[a] > x[b]; });

    int kk = std::min(K, n);
    outIdx.assign(idx.begin(), idx.begin() + kk);
}

static int top1Index(const float* x, int n) {
    int bi = 0;
    float bv = x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] > bv) { bv = x[i]; bi = i; }
    }
    return bi;
}


static void initAllOrExit(Net& model,
    const std::string& ptFile,
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
        std::cerr << "Не удалось загрузить/создать " << ptFile << "\n";
        std::exit(1);
    }

    {
        std::lock_guard<std::mutex> lk(g_trtMutex);
        if (!g_trt.initOrCreate(planFile)) {
            std::cerr << "TensorRT: не удалось инициализировать движок.\n";
            std::exit(1);
        }
        g_trtReady = true;
        g_nnBatch = TRT_MAX_BATCH;
    }

    // Первичный refit
    {
        std::scoped_lock lk(g_modelMutex, g_trtMutex);
        torch::NoGradGuard ng;

        if (!trtRefitFromTorchModel(g_trt, model)) {
            std::cerr << "TRT refit from net.pt failed at startup.\n";
        }
    }
std::cerr << "[TRT] AI_HAVE_CUDA_KERNELS=" << AI_HAVE_CUDA_KERNELS << "\n";
}

static void saveAll(const std::string& ptFile,
    const std::string& planFile,
    const std::string& optFile,
    const std::string& trainerStateFile,
    Net& model,
    Trainer& trainer) {
    {
        std::lock_guard<std::mutex> lk(g_modelMutex);

        try {
            torch::save(model, ptFile);
        }
        catch (const std::exception& e) {
            std::cerr << "torch::save(model) failed: " << e.what() << "\n";
        }

        if (!saveOptimizerState(optFile, trainer)) {
            std::cerr << "save optimizer state failed.\n";
        }

        if (!saveTrainerState(trainerStateFile, trainer)) {
            std::cerr << "save trainer state failed.\n";
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

static void safeRefitBarrier(SelfPlayContext& sp) {
    // Гарантируем, что на момент refit:
    // - server idle
    // - очередь пуста
    sp.server.waitIdle();
    sp.server.clearQueueUnsafeWhenIdle();
}



void Training(int targetGames) {
    const std::string ptFile = "net.pt";
    const std::string planFile = "net.plan";
    const std::string optFile = "optimizer.pt";
    const std::string trainerStateFile = "trainer_state.bin";

    Net model;
    initAllOrExit(model, ptFile, planFile);

    // Replay
    static constexpr size_t REPLAY_CAP = 1000000;
    ReplayBuffer rb(REPLAY_CAP);

    // Trainer: сначала восстановим steps, потом init(), потом optimizer
    Trainer trainer;

    if (loadTrainerState(trainerStateFile, trainer)) {
        std::cerr << "[Trainer] restored steps=" << trainer.steps << "\n";
    }
    else {
        std::cerr << "[Trainer] no trainer_state found, starting from step 0.\n";
    }

    trainer.init(model);

    if (loadOptimizerState(optFile, trainer)) {
        std::cerr << "[Trainer] optimizer state restored.\n";

        // После restore optimizer ещё раз принудительно выставим LR по scheduler'у
        trainer.current_lr = -1.0;
        trainer.updateLR(true);
    }
    else {
        if (trainer.steps != 0) {
            std::cerr << "[Trainer] warning: steps restored, but optimizer state not found/failed to load. "
                         "Optimizer starts fresh.\n";
        }
        else {
            std::cerr << "[Trainer] no optimizer state found, starting fresh.\n";
        }
    }

    // One self-play context for the whole training
    SelfPlayContext sp(/*nodePow2=*/(1u << 20), /*edgeCap=*/(1u << 24));
    sp.start();

    // -------------------------------
    // SCHEDULER
    // -------------------------------
    static constexpr int SELFPLAY_BLOCK_MS = 8000;
    static constexpr int MAX_GAMES_PER_BLOCK = 16;

    static constexpr int TRAIN_BLOCK_MS = 325;
    static constexpr int TRAIN_MAX_STEPS = 9999;
    static constexpr int TRAIN_WARMUP_BATCHES = 1000;

    static constexpr int REFIT_EVERY_TRAIN_BLOCKS = 8;

    const int simsPerPos = 800;
    const int maxPlies = 256;
    const bool addRootNoise = true;

    auto t0 = std::chrono::steady_clock::now();
    auto nextSave = t0 + std::chrono::hours(1);
    auto nextStat = t0 + std::chrono::seconds(10);

    int games = 0;
    int trainBlocks = 0;
    int refits = 0;

    std::cout << "Начинаем тренировку на " << targetGames << " партий...\n";

    while (games < targetGames) {
        // ===========================
        // 1) SELF-PLAY BLOCK
        // ===========================
        const auto spEnd = std::chrono::steady_clock::now() + std::chrono::milliseconds(SELFPLAY_BLOCK_MS);
        int gamesThisBlock = 0;

        while (std::chrono::steady_clock::now() < spEnd &&
            gamesThisBlock < MAX_GAMES_PER_BLOCK &&
            games < targetGames) {

            int plyCount = 0;
            bool terminated = false;

            selfPlayOneGame960(sp, rb,
                simsPerPos,
                maxPlies,
                addRootNoise,
                plyCount,
                terminated);

            ++games;
            ++gamesThisBlock;

            if (sp.T.abort.load(std::memory_order_relaxed)) {
                std::cerr << "[selfplay] MCTS aborted: oomCode=" << sp.T.oomCode.load()
                    << " -> reset table.\n";
                sp.resetForNewGame();
            }
        }

        // ===========================
        // 2) TRAIN BLOCK
        // ===========================
        safeRefitBarrier(sp);

        int didTrain = trainer.trainBlockBudgetMs(rb, model,
            TRAIN_BLOCK_MS,
            TRAIN_MAX_STEPS,
            TRAIN_WARMUP_BATCHES);

        if (didTrain > 0) {
            ++trainBlocks;
        }

        // ===========================
        // 3) REFIT TRT
        // ===========================
        if (didTrain > 0 && (trainBlocks % REFIT_EVERY_TRAIN_BLOCKS) == 0) {
            safeRefitBarrier(sp);

            std::scoped_lock lk(g_modelMutex, g_trtMutex);
            torch::NoGradGuard ng;

            if (!trtRefitFromTorchModel(g_trt, model)) {
                std::cerr << "[refit] TRT refit failed.\n";
            }
            else {
                ++refits;
            }
        }

        // ===========================
        // 4) SAVE / STATS
        // ===========================
        auto now = std::chrono::steady_clock::now();

        if (now >= nextSave) {
            safeRefitBarrier(sp);
            nextSave += std::chrono::hours(1);

            saveAll(ptFile, planFile, optFile, trainerStateFile, model, trainer);

            std::cout << "[autosave] Прогресс: " << games << " / " << targetGames << " партий.\n";
        }

        if (now >= nextStat) {
            nextStat += std::chrono::seconds(10);
            std::cerr << "[stat] games=" << games << "/" << targetGames
                << " replay=" << rb.currentSize()
                << " trainSteps=" << trainer.steps
                << " LR=" << trainer.current_lr
                << " lastLoss=" << trainer.lastLoss
                << " (P:" << trainer.lastLossP << " V:" << trainer.lastLossV << ")"
                << " nnQueue=" << sp.server.size()
                << " refits=" << refits
                << "\n";
        }
    }

    // ==========================================
    // 5) CLEAN STOP & FINAL SAVE + FINAL REBUILD
    // ==========================================
    safeRefitBarrier(sp);
    sp.stop();

    std::cout << "\n[Завершение] Собрано " << targetGames << " партий. Сохранение финальных весов...\n";
    {
        std::lock_guard<std::mutex> lk(g_modelMutex);

        try {
            torch::save(model, ptFile);
        }
        catch (const std::exception& e) {
            std::cerr << "torch::save(model) failed: " << e.what() << "\n";
        }

        if (!saveOptimizerState(optFile, trainer)) {
            std::cerr << "final save optimizer state failed.\n";
        }

        if (!saveTrainerState(trainerStateFile, trainer)) {
            std::cerr << "final save trainer state failed.\n";
        }
    }

    std::cout << "[Завершение] Запуск финальной пересборки TensorRT (Rebuild). Это займет пару минут...\n";

    // 1) shutdown + remove old plan
    {
        std::lock_guard<std::mutex> lkT(g_trtMutex);
        g_trt.shutdown();
        g_trtReady = false;
        std::remove(planFile.c_str());
    }

    // 2) rebuild/load new plan
    bool okInit = false;
    {
        std::lock_guard<std::mutex> lkT(g_trtMutex);
        okInit = g_trt.initOrCreate(planFile);
    }

    if (!okInit) {
        std::cerr << "[Завершение] FATAL ERROR: Не удалось пересобрать финальный net.plan!\n";
    }
    else {
        // 3) refit from final torch model and save final plan
        {
            std::scoped_lock lk(g_modelMutex, g_trtMutex);
            torch::NoGradGuard ng;

            if (trtRefitFromTorchModel(g_trt, model)) {
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

    std::cout << "Тренировка успешно завершена! Файлы net.pt, optimizer.pt, trainer_state.bin и net.plan готовы.\n";
}



int main() {
    const std::string ptFile = "net.pt";
    const std::string planFile = "net.plan";

    std::cout << "Введите FEN (или '960' для случайной Chess960 позиции, '-' для Training):\n";
    std::string fen;
    std::getline(std::cin, fen);

    // IMPORTANT:
    // Training initializes everything by itself.
    // Do NOT init model/TRT in main before this branch.
    if (fen == "-") {
        Training(1000000);
        return 0;
    }

    Net model;
    initAllOrExit(model, ptFile, planFile);
    if (!g_trtReady) {
        std::cout << "TensorRT движок не загружен.\n";
        return 1;
    }

    Position pos;
    std::array<uint64_t, 4> path;
    std::array<int, 64> mask;

    if (fen == "960") chess960(pos, path, mask);
    else              fenToPositionPathMask(fen, pos, path, mask);

    std::cout << "slider_backend " << (g_usePext ? "pext" : "magics") << "\n";

    MoveList ml;
    genMoves(pos, path, ml);

    std::cout << "moves " << ml.n << "\n";
    for (int i = 0; i < ml.n; ++i) {
        int m = ml.m[i];
        std::cout << sqName(m & 63) << sqName((m >> 6) & 63) << "\n";
    }

    float mctsEvalWhite = 0.5f;
    std::vector<moveState> rootMoves;
    bool rootNoise = false;

    mctsBatchedMT(pos, path, mask, 10.0, rootNoise, mctsEvalWhite, rootMoves);

    float v = 0.5f;
    std::vector<float> pol((size_t)POLICY_SIZE, 0.0f);

    g_trt.inferBatch(&pos, 1, &v, pol.data());

    std::cout << "eval=" << v << std::endl;
    std::cout << "mcts_eval_white " << mctsEvalWhite << "\n";
    std::cout << "root_moves " << rootMoves.size() << "\n";
    for (auto& ms : rootMoves) {
        int m = ms.move;
        std::cout << sqName(m & 63) << sqName((m >> 6) & 63)
            << " eval " << ms.eval
            << " visits " << ms.visits << " prior " << ms.prior << "\n";
    }

    cin.get();

    // optional clean shutdown
    {
        std::lock_guard<std::mutex> lk(g_trtMutex);
        g_trt.shutdown();
        g_trtReady = false;
    }

    return 0;
}