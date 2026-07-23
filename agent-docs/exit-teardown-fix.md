# Exit-teardown SIGSEGV — Programs must die before their Context

The test binary once printed `[ PASSED ]` and then SIGSEGV'd during process
exit (exit 139) — so ctest, CI and the Android runner all reported failure.
Device-independent (reproduced on NVIDIA, RENOIR and llvmpipe): a pure
lifetime bug, not a driver quirk.

## Root cause

`tests/launch_test.cpp` stashed every loaded program in a namespace-scope
static `std::vector<std::unique_ptr<Program>> LaunchTest::programs_` alongside
`static std::unique_ptr<Context> ctx_`. `TearDownTestSuite()` reset `ctx_` but
never cleared `programs_`, so:

1. `Context::~Context()` ran (`vkDeviceWaitIdle`, destroy pools/VMA/command
   pool, `vkDestroyDevice`, `vkDestroyInstance`) and its heap storage was freed
   while Programs still held `Context*` into it.
2. The static vector destructed during `__run_exit_handlers`, after `main`
   returned; `~ProgramImpl` dereferenced the dangling Context (`ctx->table()`,
   `ctx->device()`) and called `vkDestroyPipeline` etc. on a destroyed
   `VkDevice` → SIGSEGV.

This violated the documented invariant (a Kernel/Program must not outlive its
Program/Context) — the library RAII was correct; the fixture held state wrong.

Debugging notes worth keeping: deterministic when run directly, a heisenbug
under gdb ("exited normally" — classic UAF whose fault depends on freed-heap
layout). `MALLOC_PERTURB_=165` poisons freed memory and made it fault
deterministically under gdb with a clean backtrace.

## Fix + guard

```cpp
static void TearDownTestSuite() {
  programs_.clear();   // dependents die first, while Context/VkDevice live
  ctx_.reset();
}
```

Every fixture that stashes Context-dependent objects in statics must follow
this order (`kernel_features_test.cpp` and the examples/llm harnesses do).
Regression guard: `add_test(NAME vulkore_tests COMMAND vulkore_tests)` — ctest
fails on ANY nonzero exit, so a teardown SIGSEGV fails CI even with all
assertions green. Pass condition is "all tests pass AND the process exits 0".

## Open hardening candidate

The must-not-outlive invariant is documented, not enforced — no debug assert
against a Context alive-flag, no dependent tracking. Left minimal
deliberately; this bug is the one recorded instance of it biting.
