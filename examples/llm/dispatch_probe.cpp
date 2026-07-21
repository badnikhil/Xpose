// Is decode bandwidth-bound or dispatch-bound? Time a trivial kernel to isolate
// pure per-launch cost on this device.
#include <vulkore/vulkore.hpp>
#include <chrono>
#include <cstdio>
#include <vector>
static double now_ms(){using namespace std::chrono;
  return duration<double,std::milli>(steady_clock::now().time_since_epoch()).count();}
int main(int argc,char**argv){
  vulkore::Context ctx;
  auto prog = vulkore::Program::from_file(ctx, argv[1]);
  auto k = prog.kernel("matvec_reduce");           // trivial: cols=64, split=1
  auto out = ctx.alloc<float>(64);
  auto part= ctx.alloc<float>(64);
  std::printf("device: %s\n\n", ctx.device_name().c_str());
  for (uint32_t N : {1u, 8u, 32u, 130u, 260u}) {
    { std::vector<vulkore::Fence> w;
      for(uint32_t i=0;i<N;++i) w.push_back(vulkore::launch(k,{64u},out,part,64u,1u));
      w.back().wait(); }
    const int R=20; double t0=now_ms();
    for(int r=0;r<R;++r){
      std::vector<vulkore::Fence> fs; fs.reserve(N);
      for(uint32_t i=0;i<N;++i) fs.push_back(vulkore::launch(k,{64u},out,part,64u,1u));
      fs.back().wait();
    }
    double ms=(now_ms()-t0)/R;
    std::printf("  %4u trivial dispatches: %7.2f ms   -> %.3f ms each\n", N, ms, ms/N);
  }
  std::printf("\n  A Gemma 1B token needs ~130 matmul dispatches (26 layers x 5).\n");
  return 0;
}
