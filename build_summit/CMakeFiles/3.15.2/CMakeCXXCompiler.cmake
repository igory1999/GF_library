set(CMAKE_CXX_COMPILER "/sw/summit/llvm/9.0.0/9.0.0.patch001.cuda-10.1.168/bin/clang++")
set(CMAKE_CXX_COMPILER_ARG1 "")
set(CMAKE_CXX_COMPILER_ID "Clang")
set(CMAKE_CXX_COMPILER_VERSION "9.0.0")
set(CMAKE_CXX_COMPILER_VERSION_INTERNAL "")
set(CMAKE_CXX_COMPILER_WRAPPER "")
set(CMAKE_CXX_STANDARD_COMPUTED_DEFAULT "14")
set(CMAKE_CXX_COMPILE_FEATURES "cxx_std_98;cxx_template_template_parameters;cxx_std_11;cxx_alias_templates;cxx_alignas;cxx_alignof;cxx_attributes;cxx_auto_type;cxx_constexpr;cxx_decltype;cxx_decltype_incomplete_return_types;cxx_default_function_template_args;cxx_defaulted_functions;cxx_defaulted_move_initializers;cxx_delegating_constructors;cxx_deleted_functions;cxx_enum_forward_declarations;cxx_explicit_conversions;cxx_extended_friend_declarations;cxx_extern_templates;cxx_final;cxx_func_identifier;cxx_generalized_initializers;cxx_inheriting_constructors;cxx_inline_namespaces;cxx_lambdas;cxx_local_type_template_args;cxx_long_long_type;cxx_noexcept;cxx_nonstatic_member_init;cxx_nullptr;cxx_override;cxx_range_for;cxx_raw_string_literals;cxx_reference_qualified_functions;cxx_right_angle_brackets;cxx_rvalue_references;cxx_sizeof_member;cxx_static_assert;cxx_strong_enums;cxx_thread_local;cxx_trailing_return_types;cxx_unicode_literals;cxx_uniform_initialization;cxx_unrestricted_unions;cxx_user_literals;cxx_variadic_macros;cxx_variadic_templates;cxx_std_14;cxx_aggregate_default_initializers;cxx_attribute_deprecated;cxx_binary_literals;cxx_contextual_conversions;cxx_decltype_auto;cxx_digit_separators;cxx_generic_lambdas;cxx_lambda_init_captures;cxx_relaxed_constexpr;cxx_return_type_deduction;cxx_variable_templates;cxx_std_17;cxx_std_20")
set(CMAKE_CXX98_COMPILE_FEATURES "cxx_std_98;cxx_template_template_parameters")
set(CMAKE_CXX11_COMPILE_FEATURES "cxx_std_11;cxx_alias_templates;cxx_alignas;cxx_alignof;cxx_attributes;cxx_auto_type;cxx_constexpr;cxx_decltype;cxx_decltype_incomplete_return_types;cxx_default_function_template_args;cxx_defaulted_functions;cxx_defaulted_move_initializers;cxx_delegating_constructors;cxx_deleted_functions;cxx_enum_forward_declarations;cxx_explicit_conversions;cxx_extended_friend_declarations;cxx_extern_templates;cxx_final;cxx_func_identifier;cxx_generalized_initializers;cxx_inheriting_constructors;cxx_inline_namespaces;cxx_lambdas;cxx_local_type_template_args;cxx_long_long_type;cxx_noexcept;cxx_nonstatic_member_init;cxx_nullptr;cxx_override;cxx_range_for;cxx_raw_string_literals;cxx_reference_qualified_functions;cxx_right_angle_brackets;cxx_rvalue_references;cxx_sizeof_member;cxx_static_assert;cxx_strong_enums;cxx_thread_local;cxx_trailing_return_types;cxx_unicode_literals;cxx_uniform_initialization;cxx_unrestricted_unions;cxx_user_literals;cxx_variadic_macros;cxx_variadic_templates")
set(CMAKE_CXX14_COMPILE_FEATURES "cxx_std_14;cxx_aggregate_default_initializers;cxx_attribute_deprecated;cxx_binary_literals;cxx_contextual_conversions;cxx_decltype_auto;cxx_digit_separators;cxx_generic_lambdas;cxx_lambda_init_captures;cxx_relaxed_constexpr;cxx_return_type_deduction;cxx_variable_templates")
set(CMAKE_CXX17_COMPILE_FEATURES "cxx_std_17")
set(CMAKE_CXX20_COMPILE_FEATURES "cxx_std_20")

set(CMAKE_CXX_PLATFORM_ID "Linux")
set(CMAKE_CXX_SIMULATE_ID "")
set(CMAKE_CXX_COMPILER_FRONTEND_VARIANT "GNU")
set(CMAKE_CXX_SIMULATE_VERSION "")



set(CMAKE_AR "/usr/bin/ar")
set(CMAKE_CXX_COMPILER_AR "/sw/summit/llvm/9.0.0/9.0.0.patch001.cuda-10.1.168/bin/llvm-ar")
set(CMAKE_RANLIB "/usr/bin/ranlib")
set(CMAKE_CXX_COMPILER_RANLIB "/sw/summit/llvm/9.0.0/9.0.0.patch001.cuda-10.1.168/bin/llvm-ranlib")
set(CMAKE_LINKER "/sw/summit/xalt/1.1.4/bin/ld")
set(CMAKE_MT "")
set(CMAKE_COMPILER_IS_GNUCXX )
set(CMAKE_CXX_COMPILER_LOADED 1)
set(CMAKE_CXX_COMPILER_WORKS TRUE)
set(CMAKE_CXX_ABI_COMPILED TRUE)
set(CMAKE_COMPILER_IS_MINGW )
set(CMAKE_COMPILER_IS_CYGWIN )
if(CMAKE_COMPILER_IS_CYGWIN)
  set(CYGWIN 1)
  set(UNIX 1)
endif()

set(CMAKE_CXX_COMPILER_ENV_VAR "CXX")

if(CMAKE_COMPILER_IS_MINGW)
  set(MINGW 1)
endif()
set(CMAKE_CXX_COMPILER_ID_RUN 1)
set(CMAKE_CXX_IGNORE_EXTENSIONS inl;h;hpp;HPP;H;o;O;obj;OBJ;def;DEF;rc;RC)
set(CMAKE_CXX_SOURCE_FILE_EXTENSIONS C;M;c++;cc;cpp;cxx;mm;CPP)
set(CMAKE_CXX_LINKER_PREFERENCE 30)
set(CMAKE_CXX_LINKER_PREFERENCE_PROPAGATES 1)

# Save compiler ABI information.
set(CMAKE_CXX_SIZEOF_DATA_PTR "8")
set(CMAKE_CXX_COMPILER_ABI "ELF")
set(CMAKE_CXX_LIBRARY_ARCHITECTURE "")

if(CMAKE_CXX_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_CXX_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_CXX_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_CXX_COMPILER_ABI}")
endif()

if(CMAKE_CXX_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()

set(CMAKE_CXX_CL_SHOWINCLUDES_PREFIX "")
if(CMAKE_CXX_CL_SHOWINCLUDES_PREFIX)
  set(CMAKE_CL_SHOWINCLUDES_PREFIX "${CMAKE_CXX_CL_SHOWINCLUDES_PREFIX}")
endif()





set(CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES "/ccs/home/iyakushin/SOFT_INSTALL/llvm/9.0.0-2/adios2-2.5.0/include;/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/clang-9.0.0-2/zeromq-4.2.5-sjkln7ytr2wrfp4hig7synbkpgwgwfiy/include;/ccs/home/iyakushin/SOFT_INSTALL/llvm/9.0.0-2/bzip-1.0.6/include;/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/clang-9.0.0-2/hdf5-1.10.3-ctonnmd4sausu4frstaqtrxoisc5s3ds/include;/ccs/home/iyakushin/SOFT_INSTALL/llvm/9.0.0-2/timing_chrono-0.1/include;/autofs/nccs-svm1_sw/summit/.swci/0-core/opt/spack/20180914/linux-rhel7-ppc64le/gcc-4.8.5/papi-5.7.0-7ycgvcx5n3zk2h4dz3koxjzl72exp2le/include;/ccs/home/iyakushin/SOFT_INSTALL/llvm/9.0.0-2/sz-2.1.3/include;/ccs/home/iyakushin/SOFT_INSTALL/llvm/9.0.0-2/gsl-2.6/include;/ccs/home/iyakushin/SOFT_INSTALL/llvm/9.0.0-2/zfp-0.5.5/include;/ccs/home/iyakushin/SOFT_INSTALL/llvm/9.0.0-2/ftk-git072119/include;/ccs/home/iyakushin/SOFT_INSTALL/llvm/9.0.0-2/kokkos-git082819/include;/ccs/home/iyakushin/SOFT_INSTALL/llvm/9.0.0-2/zchecker-git072119/include;/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/clang-9.0.0-2/zlib-1.2.11-owldzme64jj2yideoqrrd4wk3r6jc5zt/include;/ccs/home/iyakushin/SOFT_INSTALL/llvm/9.0.0-2/mgard-git072019/include;/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/clang-9.0.0-2/netcdf-4.6.1-7ytljxhrqhgput6nwhijwr5o4jxuvc4q/include;/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/clang-9.0.0-2/c-blosc-1.12.1-qlmbf3gk7k7gistdhk6caf5mwgxnuryu/include;/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/clang-9.0.0-2/fftw-3.3.8-26wyqnw6lexs2a6filvjm5kzqlxfmaxo/include;/ccs/home/iyakushin/SOFT_INSTALL/llvm/9.0.0-2/libpng-1.6.37/include;/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/clang-9.0.0-2/spectrum-mpi-10.3.0.1-20190611-mivx3vscf6hfpxgjmkaoyy6ag4kh3jxh/include;/sw/summit/cuda/10.1.168/include;/sw/summit/llvm/9.0.0/9.0.0.patch001.cuda-10.1.168/include;/usr/include/c++/4.8.5;/usr/include/c++/4.8.5/ppc64le-redhat-linux;/usr/include/c++/4.8.5/backward;/autofs/nccs-svm1_sw/summit/llvm/9.0.0/9.0.0.patch001.cuda-10.1.168/lib/clang/9.0.0/include/ppc_wrappers;/usr/local/include;/autofs/nccs-svm1_sw/summit/llvm/9.0.0/9.0.0.patch001.cuda-10.1.168/lib/clang/9.0.0/include;/usr/include")
set(CMAKE_CXX_IMPLICIT_LINK_LIBRARIES "stdc++;m;gcc_s;gcc;c;gcc_s;gcc")
set(CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES "/usr/lib/gcc/ppc64le-redhat-linux/4.8.5;/usr/lib64;/lib64;/usr/lib;/autofs/nccs-svm1_sw/summit/llvm/9.0.0/9.0.0.patch001.cuda-10.1.168/lib;/lib;/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/clang-9.0.0-2/zeromq-4.2.5-sjkln7ytr2wrfp4hig7synbkpgwgwfiy/lib;/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/clang-9.0.0-2/hdf5-1.10.3-ctonnmd4sausu4frstaqtrxoisc5s3ds/lib;/autofs/nccs-svm1_sw/summit/.swci/0-core/opt/spack/20180914/linux-rhel7-ppc64le/gcc-4.8.5/papi-5.7.0-7ycgvcx5n3zk2h4dz3koxjzl72exp2le/lib;/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/clang-9.0.0-2/zlib-1.2.11-owldzme64jj2yideoqrrd4wk3r6jc5zt/lib;/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/clang-9.0.0-2/netcdf-4.6.1-7ytljxhrqhgput6nwhijwr5o4jxuvc4q/lib;/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/clang-9.0.0-2/c-blosc-1.12.1-qlmbf3gk7k7gistdhk6caf5mwgxnuryu/lib;/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/clang-9.0.0-2/fftw-3.3.8-26wyqnw6lexs2a6filvjm5kzqlxfmaxo/lib;/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/clang-9.0.0-2/spectrum-mpi-10.3.0.1-20190611-mivx3vscf6hfpxgjmkaoyy6ag4kh3jxh/lib;/sw/summit/llvm/9.0.0/9.0.0.patch001.cuda-10.1.168/lib;/sw/summit/llvm/9.0.0/9.0.0.patch001.cuda-10.1.168/lib/clang/9.0.0/lib/linux;/sw/summit/cuda/10.1.168/lib64")
set(CMAKE_CXX_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")
