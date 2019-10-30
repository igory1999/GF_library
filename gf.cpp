#include "gaussian_filter.h"

namespace GaussianFilter
{
  void generate_gaussian(double sigma, ViewMatrixType g)
  {
    size_t l = g.extent(0);
    Kokkos::parallel_for(
			 Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {l,l,l}), 
			 KOKKOS_LAMBDA (int i, int j, int k)
			 {
			   double r2 = ((l - 1)/2 - i)*((l - 1)/2 - i) +
			     ((l - 1)/2 - j)*((l - 1)/2 - j) + ((l - 1)/2 - k)*((l -1)/2 - k);
			   g(i, j, k) = exp( - r2/2/sigma/sigma)/2/sigma/sigma/M_PI;
			 }
			 );
    double sum = 0;
    Kokkos::parallel_reduce(
			    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {l,l,l}), 
			    KOKKOS_LAMBDA (int i, int j, int k, double & local_sum)
			    {
			      local_sum += g(i, j, k);
			    },
			    sum
			    );      
    Kokkos::parallel_for(
			 Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0}, {l,l,l}), 
			 KOKKOS_LAMBDA (int i, int j, int k)
			 {
			   g(i, j, k) /= sum;
			 }
			 );
  }
  
  void apply_kernel(ViewMatrixType data, ViewMatrixType result,
		    ViewMatrixConstType kernel, int t0, int t1, int t2)
  {
    int d_size = data.extent(0);
    int k_size = kernel.extent(0);
    
    int w = (k_size - 1)/2;
    Kokkos::parallel_for(
			 Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},
								{d_size, d_size, d_size},
								{t0, t1, t2}),
			 KOKKOS_LAMBDA (size_t i, size_t j, size_t k)
			 {
			   result(i, j, k) = 0.0;
			   int ip, jp, kp;
			   for(int ii = -w; ii <= w; ++ii)
			     {
			       for(int jj = -w; jj <= w;  ++jj)
				 {
				   for(int kk = -w; kk <= w; ++kk)
				     {
				       ip = i + ii;
				       jp = j + jj;
				       kp = k + kk;
				       if(ip >= 0  &&  ip < d_size &&
					  jp >= 0  &&  jp < d_size &&
					  kp >=0 && kp < d_size )
					 {
					   result(i, j, k) += kernel(w + ii, w + jj, w + kk) *
					     data(ip, jp, kp);
					 }
				     }
				 }
			     }
			 }
			 );
  }
}
