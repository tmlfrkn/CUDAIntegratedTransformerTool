set(LLVM_LINK_COMPONENTS support)
add_clang_executable(CUDAIntegratedTransformerTool
	CUDAIntegratedTransformerTool.cpp
  )
target_link_libraries(CUDAIntegratedTransformerTool
  PRIVATE
  clangLex
  clangRewrite
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangSerialization
  clangTooling
  clangEdit
  )
