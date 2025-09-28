file(REMOVE_RECURSE
  "CMakeFiles/Tracer_triad"
  "triad-instrumented"
  "full.s"
  "full.llvm"
  "triad-opt.llvm"
  "triad.llvm"
  "triad.unopt.llvm"
  "triad.obj.llvm"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/Tracer_triad.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
