#include <iostream>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
using namespace sycl;
int main()
{
    int THREADS_PER_BLOCK=2;
    std::cout << "\nRunning on " << dpct::get_default_queue().get_device().get_info<sycl::info::device::name>()
        << "\n";

    dpct::device_info deviceProp;
    std::cout << "deviceProp.get_max_compute_units() = " << deviceProp.get_max_compute_units() << std::endl;
    dpct::get_default_queue().submit([&] (handler &h){
        sycl::stream out(1024, 256, h);
        sycl::local_accessor< uint8_t , 2> dpct_local(
            sycl::range<2>( 33554432, 1), h);
                h.parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, 6/*deviceProp.get_max_compute_units()*/) *
                    sycl::range<3>(1, 1, THREADS_PER_BLOCK),
                sycl::range<3>(1, 1, THREADS_PER_BLOCK)),
            [=](sycl::nd_item<3> item_ct1){

            out <<  item_ct1 << "Hello stream!" << sycl::endl;
        } );
    dpct::get_default_queue().wait();
    } );
    return 0;
}

