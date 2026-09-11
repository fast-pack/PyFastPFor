/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 * (c) Daniel Lemire
 */

/**
 * Native ARM NEON implementations of the (small) subset of x86 SSE intrinsics
 * used by FastPFOR. This replaces the SIMDe emulation layer on aarch64 targets:
 * every operation below maps directly to native NEON instructions.
 *
 * The mappings follow the well-known SSE->NEON correspondences (the same ones
 * used by projects such as sse2neon). Shift-by-amount operations are expressed
 * with the NEON variable-shift instruction so they accept both compile-time
 * constants (which the compiler folds to immediate shifts) and runtime counts.
 *
 * This header is valid in both C99 and C++ so it can be shared by the C codecs
 * (streamvbyte.c, varintdecode.c) and the C++ headers.
 */
#ifndef FASTPFOR_NEON_H_
#define FASTPFOR_NEON_H_

#if !(defined(__aarch64__) || (defined(_MSC_VER) && defined(_M_ARM64)))
#error "fastpfor_neon.h is only for ARM (aarch64 / ARM64) targets"
#endif

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>

typedef int64x2_t __m128i;
typedef float32x4_t __m128;

/* ----------------------------- load / store ----------------------------- */

static inline __m128i _mm_loadu_si128(const __m128i *p) {
  return vld1q_s64((const int64_t *)p);
}
static inline __m128i _mm_lddqu_si128(const __m128i *p) {
  return vld1q_s64((const int64_t *)p);
}
static inline __m128i _mm_load_si128(const __m128i *p) {
  return vld1q_s64((const int64_t *)p);
}
static inline __m128i _mm_loadl_epi64(const __m128i *p) {
  return vcombine_s64(vld1_s64((const int64_t *)p), vdup_n_s64(0));
}
static inline void _mm_storeu_si128(__m128i *p, __m128i a) {
  vst1q_s64((int64_t *)p, a);
}
static inline void _mm_store_si128(__m128i *p, __m128i a) {
  vst1q_s64((int64_t *)p, a);
}
static inline void _mm_stream_si128(__m128i *p, __m128i a) {
  vst1q_s64((int64_t *)p, a);
}
static inline void _mm_storel_epi64(__m128i *p, __m128i a) {
  vst1_s64((int64_t *)p, vget_low_s64(a));
}

/* --------------------------------- set ---------------------------------- */

static inline __m128i _mm_setzero_si128(void) { return vdupq_n_s64(0); }
static inline __m128i _mm_set1_epi32(int a) {
  return vreinterpretq_s64_s32(vdupq_n_s32(a));
}
static inline __m128i _mm_set1_epi16(short a) {
  return vreinterpretq_s64_s16(vdupq_n_s16(a));
}
static inline __m128i _mm_set1_epi8(signed char a) {
  return vreinterpretq_s64_s8(vdupq_n_s8(a));
}
static inline __m128i _mm_set_epi64x(int64_t e1, int64_t e0) {
  int64_t d[2];
  d[0] = e0;
  d[1] = e1;
  return vld1q_s64(d);
}
static inline __m128i
_mm_set_epi8(signed char e15, signed char e14, signed char e13, signed char e12,
             signed char e11, signed char e10, signed char e9, signed char e8,
             signed char e7, signed char e6, signed char e5, signed char e4,
             signed char e3, signed char e2, signed char e1, signed char e0) {
  int8_t d[16];
  d[0] = e0;   d[1] = e1;   d[2] = e2;   d[3] = e3;
  d[4] = e4;   d[5] = e5;   d[6] = e6;   d[7] = e7;
  d[8] = e8;   d[9] = e9;   d[10] = e10; d[11] = e11;
  d[12] = e12; d[13] = e13; d[14] = e14; d[15] = e15;
  return vreinterpretq_s64_s8(vld1q_s8(d));
}
static inline __m128i
_mm_setr_epi8(signed char e0, signed char e1, signed char e2, signed char e3,
              signed char e4, signed char e5, signed char e6, signed char e7,
              signed char e8, signed char e9, signed char e10, signed char e11,
              signed char e12, signed char e13, signed char e14,
              signed char e15) {
  int8_t d[16];
  d[0] = e0;   d[1] = e1;   d[2] = e2;   d[3] = e3;
  d[4] = e4;   d[5] = e5;   d[6] = e6;   d[7] = e7;
  d[8] = e8;   d[9] = e9;   d[10] = e10; d[11] = e11;
  d[12] = e12; d[13] = e13; d[14] = e14; d[15] = e15;
  return vreinterpretq_s64_s8(vld1q_s8(d));
}
static inline __m128i _mm_setr_epi16(short e0, short e1, short e2, short e3,
                                     short e4, short e5, short e6, short e7) {
  int16_t d[8];
  d[0] = e0; d[1] = e1; d[2] = e2; d[3] = e3;
  d[4] = e4; d[5] = e5; d[6] = e6; d[7] = e7;
  return vreinterpretq_s64_s16(vld1q_s16(d));
}

/* ----------------------------- bitwise / arith -------------------------- */

static inline __m128i _mm_and_si128(__m128i a, __m128i b) {
  return vandq_s64(a, b);
}
static inline __m128i _mm_or_si128(__m128i a, __m128i b) {
  return vorrq_s64(a, b);
}
static inline __m128i _mm_add_epi32(__m128i a, __m128i b) {
  return vreinterpretq_s64_s32(
      vaddq_s32(vreinterpretq_s32_s64(a), vreinterpretq_s32_s64(b)));
}
static inline __m128i _mm_sub_epi32(__m128i a, __m128i b) {
  return vreinterpretq_s64_s32(
      vsubq_s32(vreinterpretq_s32_s64(a), vreinterpretq_s32_s64(b)));
}
static inline __m128i _mm_mullo_epi32(__m128i a, __m128i b) {
  return vreinterpretq_s64_s32(
      vmulq_s32(vreinterpretq_s32_s64(a), vreinterpretq_s32_s64(b)));
}
static inline __m128i _mm_mullo_epi16(__m128i a, __m128i b) {
  return vreinterpretq_s64_s16(
      vmulq_s16(vreinterpretq_s16_s64(a), vreinterpretq_s16_s64(b)));
}

/* --------------------------------- shifts ------------------------------- */
/* Variable-shift form: accepts runtime counts; the compiler lowers a
 * constant count to a native immediate shift. A right shift is a left shift
 * by a negative amount (NEON semantics); counts >= element width yield 0,
 * matching SSE. */

static inline __m128i _mm_slli_epi32(__m128i a, int imm) {
  return vreinterpretq_s64_u32(
      vshlq_u32(vreinterpretq_u32_s64(a), vdupq_n_s32(imm)));
}
static inline __m128i _mm_srli_epi32(__m128i a, int imm) {
  return vreinterpretq_s64_u32(
      vshlq_u32(vreinterpretq_u32_s64(a), vdupq_n_s32(-imm)));
}
static inline __m128i _mm_srli_epi16(__m128i a, int imm) {
  return vreinterpretq_s64_u16(
      vshlq_u16(vreinterpretq_u16_s64(a), vdupq_n_s16((int16_t)-imm)));
}
static inline __m128i _mm_slli_epi64(__m128i a, int imm) {
  return vreinterpretq_s64_u64(
      vshlq_u64(vreinterpretq_u64_s64(a), vdupq_n_s64(imm)));
}
static inline __m128i _mm_srli_epi64(__m128i a, int imm) {
  return vreinterpretq_s64_u64(
      vshlq_u64(vreinterpretq_u64_s64(a), vdupq_n_s64(-imm)));
}

/* Whole-register byte shifts. The byte count is always a compile-time
 * constant in FastPFOR, so vextq_u8 (which needs an immediate) is used. */
#define _mm_srli_si128(a, imm)                                                 \
  vreinterpretq_s64_u8(                                                        \
      vextq_u8(vreinterpretq_u8_s64(a), vdupq_n_u8(0), (imm)))
#define _mm_slli_si128(a, imm)                                                 \
  vreinterpretq_s64_u8(                                                        \
      vextq_u8(vdupq_n_u8(0), vreinterpretq_u8_s64(a), (16 - (imm))))

/* ------------------------------- compares ------------------------------- */

static inline __m128i _mm_cmplt_epi32(__m128i a, __m128i b) {
  return vreinterpretq_s64_u32(
      vcltq_s32(vreinterpretq_s32_s64(a), vreinterpretq_s32_s64(b)));
}
static inline __m128i _mm_cmpeq_epi8(__m128i a, __m128i b) {
  return vreinterpretq_s64_u8(
      vceqq_u8(vreinterpretq_u8_s64(a), vreinterpretq_u8_s64(b)));
}

/* ------------------------------- shuffles ------------------------------- */

static inline __m128i _mm_shuffle_epi8(__m128i a, __m128i b) {
  /* pshufb: a byte of the index with its high bit set produces 0. Masking the
   * index with 0x8F leaves the low nibble and the high bit; vqtbl1q_u8 then
   * yields 0 for any index >= 16 (i.e. when the high bit was set). */
  uint8x16_t tbl = vreinterpretq_u8_s64(a);
  uint8x16_t idx = vandq_u8(vreinterpretq_u8_s64(b), vdupq_n_u8(0x8F));
  return vreinterpretq_s64_u8(vqtbl1q_u8(tbl, idx));
}
static inline __m128i _mm_shuffle_epi32(__m128i a, const int imm) {
  /* Permute the four 32-bit lanes. `imm` is a compile-time constant at every
   * call site, so the broadcast fast paths fold away and the general case
   * builds a constant byte index that compiles to a single TBL. */
  uint32x4_t v = vreinterpretq_u32_s64(a);
  if (imm == 0x00)
    return vreinterpretq_s64_u32(vdupq_laneq_u32(v, 0));
  if (imm == 0x55)
    return vreinterpretq_s64_u32(vdupq_laneq_u32(v, 1));
  if (imm == 0xAA)
    return vreinterpretq_s64_u32(vdupq_laneq_u32(v, 2));
  if (imm == 0xFF)
    return vreinterpretq_s64_u32(vdupq_laneq_u32(v, 3));
  {
    const uint8_t b0 = (uint8_t)((imm & 3) * 4);
    const uint8_t b1 = (uint8_t)(((imm >> 2) & 3) * 4);
    const uint8_t b2 = (uint8_t)(((imm >> 4) & 3) * 4);
    const uint8_t b3 = (uint8_t)(((imm >> 6) & 3) * 4);
    const uint8x16_t idx = {
        b0, (uint8_t)(b0 + 1), (uint8_t)(b0 + 2), (uint8_t)(b0 + 3),
        b1, (uint8_t)(b1 + 1), (uint8_t)(b1 + 2), (uint8_t)(b1 + 3),
        b2, (uint8_t)(b2 + 1), (uint8_t)(b2 + 2), (uint8_t)(b2 + 3),
        b3, (uint8_t)(b3 + 1), (uint8_t)(b3 + 2), (uint8_t)(b3 + 3)};
    return vreinterpretq_s64_u8(vqtbl1q_u8(vreinterpretq_u8_s64(a), idx));
  }
}
static inline __m128i _mm_blend_epi16(__m128i a, __m128i b, const int imm) {
  /* Per-16-bit-lane select from `a` (0) or `b` (1). `imm` is constant at every
   * call site, so this compound-literal mask folds to a constant vector load
   * feeding a single BSL, with no stack round-trip. */
  const uint16x8_t mask = {
      (imm & 0x01) ? (uint16_t)0xFFFF : (uint16_t)0,
      (imm & 0x02) ? (uint16_t)0xFFFF : (uint16_t)0,
      (imm & 0x04) ? (uint16_t)0xFFFF : (uint16_t)0,
      (imm & 0x08) ? (uint16_t)0xFFFF : (uint16_t)0,
      (imm & 0x10) ? (uint16_t)0xFFFF : (uint16_t)0,
      (imm & 0x20) ? (uint16_t)0xFFFF : (uint16_t)0,
      (imm & 0x40) ? (uint16_t)0xFFFF : (uint16_t)0,
      (imm & 0x80) ? (uint16_t)0xFFFF : (uint16_t)0};
  return vreinterpretq_s64_u16(vbslq_u16(mask, vreinterpretq_u16_s64(b),
                                         vreinterpretq_u16_s64(a)));
}

/* --------------------------- extract / convert -------------------------- */

#define _mm_extract_epi32(a, imm)                                              \
  vgetq_lane_s32(vreinterpretq_s32_s64(a), (imm))

static inline int _mm_cvtsi128_si32(__m128i a) {
  return vgetq_lane_s32(vreinterpretq_s32_s64(a), 0);
}
static inline __m128i _mm_cvtepu8_epi16(__m128i a) {
  return vreinterpretq_s64_u16(vmovl_u8(vget_low_u8(vreinterpretq_u8_s64(a))));
}
static inline __m128i _mm_cvtepu16_epi32(__m128i a) {
  return vreinterpretq_s64_u32(
      vmovl_u16(vget_low_u16(vreinterpretq_u16_s64(a))));
}
static inline __m128i _mm_cvtepi8_epi32(__m128i a) {
  int16x8_t t16 = vmovl_s8(vget_low_s8(vreinterpretq_s8_s64(a)));
  return vreinterpretq_s64_s32(vmovl_s16(vget_low_s16(t16)));
}

/* -------------------------------- masks --------------------------------- */

static inline int _mm_movemask_epi8(__m128i a) {
  uint8x16_t input = vreinterpretq_u8_s64(a);
  uint16x8_t high_bits = vreinterpretq_u16_u8(vshrq_n_u8(input, 7));
  uint32x4_t paired16 =
      vreinterpretq_u32_u16(vsraq_n_u16(high_bits, high_bits, 7));
  uint64x2_t paired32 =
      vreinterpretq_u64_u32(vsraq_n_u32(paired16, paired16, 14));
  uint8x16_t paired64 =
      vreinterpretq_u8_u64(vsraq_n_u64(paired32, paired32, 28));
  return vgetq_lane_u8(paired64, 0) | ((int)vgetq_lane_u8(paired64, 8) << 8);
}

/* --------------------------------- float -------------------------------- */

static inline __m128 _mm_castsi128_ps(__m128i a) {
  return vreinterpretq_f32_s64(a);
}
static inline int _mm_movemask_ps(__m128 a) {
  static const int32_t shifts[4] = {0, 1, 2, 3};
  uint32x4_t signs = vshrq_n_u32(vreinterpretq_u32_f32(a), 31);
  uint32x4_t weighted = vshlq_u32(signs, vld1q_s32(shifts));
  return (int)vaddvq_u32(weighted);
}

#endif /* FASTPFOR_NEON_H_ */
