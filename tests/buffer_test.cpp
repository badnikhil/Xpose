#include <vulkore/vulkore.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <vector>

namespace {

// One Context for the whole suite (device selected via VULKORE_DEVICE by the
// test-matrix runner).
class BufferTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { ctx_ = std::make_unique<vulkore::Context>(); }
  static void TearDownTestSuite() { ctx_.reset(); }
  static std::unique_ptr<vulkore::Context> ctx_;
};
std::unique_ptr<vulkore::Context> BufferTest::ctx_;

// A 16-byte POD (sizeof % 4 == 0, so it satisfies alloc<T>'s static_assert),
// used to prove typed alloc<T> sizing works for element types wider than u32.
struct Vec4 {
  float x, y, z, w;
};
static_assert(sizeof(Vec4) == 16, "Vec4 must be 16 bytes for this test");

std::vector<uint32_t> pattern_u32(size_t n, uint32_t salt) {
  std::vector<uint32_t> v(n);
  for (size_t i = 0; i < n; ++i) {
    v[i] = static_cast<uint32_t>(i) * 2654435761u ^ salt;
  }
  return v;
}

// Deterministic per-position byte pattern; different salts give byte streams
// that differ at every position, so a patched window is distinguishable from
// the surrounding base bytes byte-for-byte.
uint8_t byte_at(VkDeviceSize i, uint8_t salt) {
  return static_cast<uint8_t>((i * 131u + 17u) ^ salt);
}

void roundtrip(vulkore::Context& ctx, size_t count, vulkore::Usage usage,
               bool force_staging) {
  SCOPED_TRACE("count=" + std::to_string(count) +
               " usage=" + (usage == vulkore::Usage::DeviceLocal ? "DeviceLocal" : "HostVisible") +
               " force_staging=" + (force_staging ? "yes" : "no"));
  vulkore::Buffer buf = ctx.alloc<uint32_t>(count, usage);
  buf.set_force_staging(force_staging);
  ASSERT_TRUE(buf.valid());
  ASSERT_EQ(buf.size_bytes(), count * sizeof(uint32_t));

  const auto src = pattern_u32(count, 0xA5A5A5A5u);
  buf.upload(std::span<const uint32_t>(src));

  std::vector<uint32_t> dst(count, 0);
  buf.download(std::span<uint32_t>(dst));
  ASSERT_EQ(src, dst);
}

// Element counts deliberately include non-multiples of 4 (odd/prime counts);
// all sizes stay 4-byte-typed as storage buffers require.
const size_t kCounts[] = {1, 3, 7, 64, 1023, 4096, 100003};

TEST_F(BufferTest, RoundtripDeviceLocal) {
  for (size_t n : kCounts) roundtrip(*ctx_, n, vulkore::Usage::DeviceLocal, false);
}

TEST_F(BufferTest, RoundtripHostVisible) {
  for (size_t n : kCounts) roundtrip(*ctx_, n, vulkore::Usage::HostVisible, false);
}

// Forcing the staging path guarantees the one-shot vkCmdCopyBuffer transfer is
// exercised even on devices where every allocation is host-visible (lavapipe,
// UMA iGPU).
TEST_F(BufferTest, RoundtripForcedStagingDeviceLocal) {
  for (size_t n : kCounts) roundtrip(*ctx_, n, vulkore::Usage::DeviceLocal, true);
}

TEST_F(BufferTest, RoundtripForcedStagingHostVisible) {
  for (size_t n : kCounts) roundtrip(*ctx_, n, vulkore::Usage::HostVisible, true);
}

TEST_F(BufferTest, HostVisibleUsageIsMappable) {
  vulkore::Buffer buf = ctx_->alloc<uint32_t>(64, vulkore::Usage::HostVisible);
  EXPECT_TRUE(buf.host_visible());
}

TEST_F(BufferTest, FloatRoundtrip) {
  const size_t n = 1024;
  std::vector<float> src(n);
  for (size_t i = 0; i < n; ++i) src[i] = 0.5f * static_cast<float>(i) - 100.0f;
  src[0] = -0.0f;                                   // IEEE edge inputs survive
  src[1] = std::numeric_limits<float>::quiet_NaN(); // transfers bit-exactly
  src[2] = std::numeric_limits<float>::infinity();

  vulkore::Buffer buf = ctx_->alloc<float>(n);
  buf.upload(std::span<const float>(src));
  std::vector<float> dst(n, 0.0f);
  buf.download(std::span<float>(dst));
  // Compare bit patterns (NaN != NaN would fail a value compare).
  EXPECT_EQ(0, std::memcmp(src.data(), dst.data(), n * sizeof(float)));
}

TEST_F(BufferTest, PartialUploadDownloadAtOffset) {
  const size_t n = 256;
  const auto base = pattern_u32(n, 0x11111111u);
  vulkore::Buffer buf = ctx_->alloc<uint32_t>(n);
  buf.upload(std::span<const uint32_t>(base));

  // Overwrite the middle quarter at a byte offset.
  const size_t off_elems = 64, sub_elems = 64;
  const auto patch = pattern_u32(sub_elems, 0xDEADBEEFu);
  buf.upload(std::span<const uint32_t>(patch), off_elems * sizeof(uint32_t));

  std::vector<uint32_t> got(n, 0);
  buf.download(std::span<uint32_t>(got));
  for (size_t i = 0; i < n; ++i) {
    const uint32_t want = (i >= off_elems && i < off_elems + sub_elems)
                              ? patch[i - off_elems]
                              : base[i];
    ASSERT_EQ(got[i], want) << "at element " << i;
  }

  // Partial download of just the patched window.
  std::vector<uint32_t> window(sub_elems, 0);
  buf.download(std::span<uint32_t>(window), off_elems * sizeof(uint32_t));
  EXPECT_EQ(window, patch);
}

TEST_F(BufferTest, MoveTransfersOwnership) {
  vulkore::Buffer a = ctx_->alloc<uint32_t>(16);
  const auto data = pattern_u32(16, 0x22222222u);
  a.upload(std::span<const uint32_t>(data));

  vulkore::Buffer b = std::move(a);
  EXPECT_FALSE(a.valid());  // NOLINT(bugprone-use-after-move) — testing moved-from state
  ASSERT_TRUE(b.valid());
  std::vector<uint32_t> got(16, 0);
  b.download(std::span<uint32_t>(got));
  EXPECT_EQ(got, data);

  vulkore::Buffer c;
  c = std::move(b);
  EXPECT_FALSE(b.valid());  // NOLINT(bugprone-use-after-move)
  ASSERT_TRUE(c.valid());
  got.assign(16, 0);
  c.download(std::span<uint32_t>(got));
  EXPECT_EQ(got, data);
}

TEST_F(BufferTest, OutOfBoundsTransferThrows) {
  vulkore::Buffer buf = ctx_->alloc<uint32_t>(4);
  std::vector<uint32_t> big(5, 0);
  EXPECT_THROW(buf.upload(std::span<const uint32_t>(big)), vulkore::Error);
  EXPECT_THROW(buf.download(std::span<uint32_t>(big)), vulkore::Error);
  std::vector<uint32_t> ok(4, 0);
  EXPECT_THROW(buf.download(std::span<uint32_t>(ok), 4), vulkore::Error);
}

TEST_F(BufferTest, ZeroSizeAllocThrows) {
  EXPECT_THROW(ctx_->alloc<uint32_t>(0), vulkore::Error);
}

// PartialUploadDownloadAtOffset above only runs the memcpy path on UMA devices.
// Forcing set_force_staging(true) drives the SAME offset upload/download
// through the vkCmdCopyBuffer staging path, exercising its region offset
// arithmetic (buffer.cpp:122 `VkBufferCopy{0, dst_offset, size}` on upload and
// buffer.cpp:142 `VkBufferCopy{src_offset, 0, size}` on download) — otherwise
// dead code on any host-visible/UMA allocation. Both Usage kinds are covered.
TEST_F(BufferTest, ForcedStagingPartialOffsetRoundtrip) {
  for (vulkore::Usage usage : {vulkore::Usage::DeviceLocal, vulkore::Usage::HostVisible}) {
    SCOPED_TRACE(usage == vulkore::Usage::DeviceLocal ? "DeviceLocal" : "HostVisible");
    const size_t n = 256;
    const auto base = pattern_u32(n, 0x33333333u);
    vulkore::Buffer buf = ctx_->alloc<uint32_t>(n, usage);
    buf.set_force_staging(true);  // force the copy path even on UMA
    buf.upload(std::span<const uint32_t>(base));

    const size_t off_elems = 64, sub_elems = 64;
    const auto patch = pattern_u32(sub_elems, 0xCAFEBABEu);
    buf.upload(std::span<const uint32_t>(patch), off_elems * sizeof(uint32_t));

    std::vector<uint32_t> got(n, 0);
    buf.download(std::span<uint32_t>(got));
    for (size_t i = 0; i < n; ++i) {
      const uint32_t want = (i >= off_elems && i < off_elems + sub_elems)
                                ? patch[i - off_elems]
                                : base[i];
      ASSERT_EQ(got[i], want) << "at element " << i;
    }

    // Partial download of just the patched window via the staging src_offset path.
    std::vector<uint32_t> window(sub_elems, 0);
    buf.download(std::span<uint32_t>(window), off_elems * sizeof(uint32_t));
    EXPECT_EQ(window, patch);
  }
}

// Move-assigning over a buffer that already owns a live allocation must destroy
// the old resource before adopting the source (buffer.cpp:95 destroy() inside
// operator=). MoveTransfersOwnership only ever assigns into an EMPTY target, so
// this is the only coverage of the destroy-old branch.
TEST_F(BufferTest, MoveAssignOverLiveBufferReleasesOld) {
  vulkore::Buffer a = ctx_->alloc<uint32_t>(32);
  vulkore::Buffer b = ctx_->alloc<uint32_t>(48);
  const auto data_a = pattern_u32(32, 0x0A0A0A0Au);
  const auto data_b = pattern_u32(48, 0x0B0B0B0Bu);
  a.upload(std::span<const uint32_t>(data_a));
  b.upload(std::span<const uint32_t>(data_b));

  a = std::move(b);  // destroys a's old (live) 32-elem allocation, adopts b's

  EXPECT_FALSE(b.valid());  // NOLINT(bugprone-use-after-move) — testing moved-from
  ASSERT_TRUE(a.valid());
  ASSERT_EQ(a.size_bytes(), 48u * sizeof(uint32_t));  // now holds b's buffer
  std::vector<uint32_t> got(48, 0);
  a.download(std::span<uint32_t>(got));
  EXPECT_EQ(got, data_b);
}

// A zero-size transfer short-circuits before any bounds check or Vulkan work
// (buffer.cpp:107 upload / :128 download) — it is a no-op, NOT an error (unlike
// alloc(0), which throws). Even an offset at the very end of the buffer is fine
// when the size is 0.
TEST_F(BufferTest, ZeroSizeTransferIsNoOp) {
  vulkore::Buffer buf = ctx_->alloc<uint32_t>(8);
  const auto data = pattern_u32(8, 0x77777777u);
  buf.upload(std::span<const uint32_t>(data));

  uint32_t scratch = 0xFFFFFFFFu;
  EXPECT_NO_THROW(buf.upload_bytes(&scratch, 0, 0));
  EXPECT_NO_THROW(buf.download_bytes(&scratch, 0, 0));
  // offset == size_ with size 0 returns before the OOB check, so it must not throw.
  EXPECT_NO_THROW(buf.upload_bytes(&scratch, 0, buf.size_bytes()));
  EXPECT_NO_THROW(buf.download_bytes(&scratch, 0, buf.size_bytes()));
  EXPECT_EQ(scratch, 0xFFFFFFFFu);  // a 0-byte download left scratch untouched

  std::vector<uint32_t> got(8, 0);
  buf.download(std::span<uint32_t>(got));
  EXPECT_EQ(got, data);  // original contents intact
}

// alloc<T>(n) for a 16-byte POD sizes the buffer as n * sizeof(T) (buffer.hpp:77)
// and satisfies the sizeof(T) % 4 == 0 static_assert (buffer.hpp:75). A full
// roundtrip confirms the byte arithmetic holds for element types wider than u32.
TEST_F(BufferTest, TypedAllocSizeForWideType) {
  const size_t n = 100;
  vulkore::Buffer buf = ctx_->alloc<Vec4>(n);
  EXPECT_EQ(buf.size_bytes(), n * sizeof(Vec4));

  std::vector<Vec4> src(n);
  for (size_t i = 0; i < n; ++i) {
    const float f = static_cast<float>(i);
    src[i] = {f, f * 2.0f, -f, 0.5f * f};
  }
  buf.upload(std::span<const Vec4>(src));
  std::vector<Vec4> dst(n);
  buf.download(std::span<Vec4>(dst));
  EXPECT_EQ(0, std::memcmp(src.data(), dst.data(), n * sizeof(Vec4)));
}

// One unaligned/sub-atom (offset,size) round-trip, in one of two directions:
//   - upload variant:   base-fill the whole buffer, patch [off,off+sz) via an
//     unaligned upload_bytes, then read it all back and assert the window holds
//     the patch AND every byte outside the window is still the base value.
//   - download variant: fill the buffer with a known stream, download JUST the
//     [off,off+sz) window into the middle of a sentinel-bracketed scratch, and
//     assert the window bytes are exact AND the destination guard bytes on
//     either side were untouched (no over-read/under-read on the copy).
// Byte-granular so it catches a partial-drop of neighbouring bytes. On
// non-coherent memory (Mali HOST_CACHED) this truly exercises the VMA
// flush/invalidate range-rounding; on coherent desktop memory it passes as a
// no-op (flush/invalidate are VMA no-ops there).
void unaligned_case(vulkore::Buffer& buf, VkDeviceSize bufsz, VkDeviceSize off,
                    VkDeviceSize sz, bool download_variant) {
  if (!download_variant) {
    std::vector<uint8_t> base(bufsz);
    for (VkDeviceSize i = 0; i < bufsz; ++i) base[i] = byte_at(i, 0x11);
    buf.upload_bytes(base.data(), bufsz, 0);

    std::vector<uint8_t> patch(sz);
    for (VkDeviceSize j = 0; j < sz; ++j) patch[j] = byte_at(off + j, 0xE7);
    buf.upload_bytes(patch.data(), sz, off);

    std::vector<uint8_t> got(bufsz, 0xAB);
    buf.download_bytes(got.data(), bufsz, 0);
    for (VkDeviceSize i = 0; i < bufsz; ++i) {
      const uint8_t want = (i >= off && i < off + sz) ? patch[i - off] : base[i];
      ASSERT_EQ(got[i], want) << "byte " << i
                              << (i >= off && i < off + sz ? " (in window)"
                                                           : " (surrounding)");
    }
  } else {
    std::vector<uint8_t> known(bufsz);
    for (VkDeviceSize i = 0; i < bufsz; ++i) known[i] = byte_at(i, 0x5C);
    buf.upload_bytes(known.data(), bufsz, 0);

    const VkDeviceSize guard = 4;
    std::vector<uint8_t> dst(sz + 2 * guard, 0xEE);  // sentinel brackets
    buf.download_bytes(dst.data() + guard, sz, off);
    for (VkDeviceSize j = 0; j < guard; ++j) {
      ASSERT_EQ(dst[j], 0xEE) << "pre-guard clobbered at " << j;
      ASSERT_EQ(dst[guard + sz + j], 0xEE) << "post-guard clobbered at " << j;
    }
    for (VkDeviceSize j = 0; j < sz; ++j) {
      ASSERT_EQ(dst[guard + j], known[off + j]) << "window byte " << j;
    }
  }
}

// Regression guard for the 2026-07-18 Mali coherency fix (mali-coherency-fix.md).
// Sub-nonCoherentAtomSize / unaligned partial transfers at an offset across
// {HostVisible, DeviceLocal} x {direct, force_staging} x {upload-at-offset,
// download-at-offset}. The (offset,size) matrix is deliberately sub-atom and
// atom-boundary-crossing; on a real non-coherent device (Mali HOST_CACHED) this
// stresses the flush/invalidate range-rounding on both the direct and staging
// paths, on desktop coherent memory it is a no-op pass. Byte-exact + surrounding
// bytes must be untouched. Do NOT hardcode 64 — the sub-atom offsets/sizes are
// sub-atom on any real atom size, and one boundary-crossing pair is derived from
// the queried nonCoherentAtomSize.
TEST_F(BufferTest, BufferUnalignedSubAtomRoundtrip) {
  const VkDeviceSize atom = ctx_->device_properties().limits.nonCoherentAtomSize;
  const VkDeviceSize bufsz = 1024;  // fits every pair below for any atom <= 256

  struct Pair {
    VkDeviceSize off, sz;
    const char* tag;
  };
  std::vector<Pair> pairs = {
      {1, 3, "off1 sz3 (sub-atom)"},
      {60, 8, "off60 sz8 (crosses 64B line)"},
      {63, 2, "off63 sz2 (crosses 64B line)"},
      {192, 65, "off192 sz65 (multi-atom, unaligned size)"},
      {0, 64, "off0 sz64 (aligned control)"},
      {128, 64, "off128 sz64 (aligned control)"},
  };
  // A boundary-crossing pair derived from the REAL atom size (portable to any
  // nonCoherentAtomSize): starts one byte before an atom line and spans past it.
  if (atom >= 2 && (atom - 1) + (atom + 3) <= bufsz) {
    pairs.push_back({atom - 1, atom + 3, "atom-1 .. crosses atom line"});
  }

  for (vulkore::Usage usage :
       {vulkore::Usage::HostVisible, vulkore::Usage::DeviceLocal}) {
    for (bool force_staging : {false, true}) {
      SCOPED_TRACE(std::string("usage=") +
                   (usage == vulkore::Usage::DeviceLocal ? "DeviceLocal"
                                                       : "HostVisible") +
                   " force_staging=" + (force_staging ? "yes" : "no"));
      vulkore::Buffer buf = ctx_->alloc_bytes(bufsz, usage);
      buf.set_force_staging(force_staging);
      ASSERT_TRUE(buf.valid());
      for (const Pair& p : pairs) {
        ASSERT_LE(p.off + p.sz, bufsz);
        for (bool download_variant : {false, true}) {
          SCOPED_TRACE(std::string("pair=") + p.tag +
                       " variant=" + (download_variant ? "download" : "upload"));
          unaligned_case(buf, bufsz, p.off, p.sz, download_variant);
        }
      }
    }
  }
}

// Locks the OOM XP_CHECK path in Buffer allocation (buffer.cpp:68-69). Asks for
// a device-local allocation larger than any heap can back: a conforming driver
// rejects it in vkAllocateMemory WITHOUT committing any backing pages (fails
// fast, zero host-RAM pressure). We never write to it, so no real memory is
// faulted in. Non-flaky by design: if a driver lazily SUCCEEDS instead
// (over-committing malloc on a system-RAM heap), we free it immediately and
// GTEST_SKIP rather than risk OOM-killing the host.
//
// Sizing note: the task's data point is the largest DEVICE_LOCAL heap, but VMA
// (AUTO_PREFER_DEVICE) will fall back to a LARGER host-visible heap if the
// device-local one can't fit — on NVIDIA that means a heap+1GiB request spills
// into the 10 GiB system-RAM heap and actually COMMITS ~5 GiB (observed 11 s,
// real OOM risk on a low-RAM box). So we size the request past EVERY heap,
// leaving VMA no fallback target — the allocation then fails fast everywhere.
TEST_F(BufferTest, OverHeapAllocThrowsCleanly) {
  VkPhysicalDeviceMemoryProperties mem{};
  vkGetPhysicalDeviceMemoryProperties(ctx_->physical_device(), &mem);
  VkDeviceSize largest_device_local = 0;  // the task's queried data point
  VkDeviceSize largest_any_heap = 0;      // used to defeat VMA's cross-heap fallback
  for (uint32_t i = 0; i < mem.memoryHeapCount; ++i) {
    const VkDeviceSize sz = mem.memoryHeaps[i].size;
    if (sz > largest_any_heap) largest_any_heap = sz;
    if ((mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) &&
        sz > largest_device_local) {
      largest_device_local = sz;
    }
  }
  ASSERT_GT(largest_device_local, 0u)
      << "device must expose at least one device-local heap";

  // Largest heap + 1 GiB: over EVERY heap, so a single vkAllocateMemory of this
  // size fails without any backing store on a conforming driver (no fallback).
  const VkDeviceSize over = largest_any_heap + (VkDeviceSize(1) << 30);
  ASSERT_GT(over, largest_any_heap) << "over-heap size computation overflowed";
  const size_t count = static_cast<size_t>(over / sizeof(uint32_t));

  try {
    vulkore::Buffer huge = ctx_->alloc<uint32_t>(count, vulkore::Usage::DeviceLocal);
    // Driver lazily satisfied an over-heap request (legal on system-RAM heaps
    // that over-commit). Release it before doing ANYTHING else, then skip — we
    // must never touch/use it (that could fault in > heap bytes and OOM-kill).
    huge = vulkore::Buffer{};  // free immediately via move-assign over the live one
    GTEST_SKIP() << "driver '" << ctx_->device_name()
                 << "' lazily satisfied an over-heap allocation (largest heap="
                 << largest_any_heap << " B, device-local=" << largest_device_local
                 << " B, asked " << over
                 << " B) without throwing; released, nothing to assert";
  } catch (const vulkore::Error& e) {
    // The throw itself is the contract. Its VkResult is ideally
    // OUT_OF_DEVICE_MEMORY; accept OUT_OF_HOST_MEMORY too (some drivers report
    // the host-side failure for a system-RAM heap).
    const VkResult r = e.result();
    EXPECT_TRUE(r == VK_ERROR_OUT_OF_DEVICE_MEMORY ||
                r == VK_ERROR_OUT_OF_HOST_MEMORY)
        << "over-heap alloc threw, but with an unexpected VkResult; what()="
        << e.what();
  }
}

}  // namespace
