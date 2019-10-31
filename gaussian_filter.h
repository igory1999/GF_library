#include <iostream>
#include <cmath>
#include <Kokkos_Core.hpp>

namespace GaussianFilter
{
  typedef Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::Cuda>  ViewMatrixType;
  typedef Kokkos::View<const double***, Kokkos::MemoryTraits<Kokkos::RandomAccess>> ViewMatrixConstType;

  void generate_gaussian(double sigma, ViewMatrixType g);
  void apply_kernel(ViewMatrixType data, ViewMatrixType result, 
		    ViewMatrixConstType kernel, int t0, int t1, int t2);
}
