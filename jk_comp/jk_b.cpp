#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <lawrap/blas.h>
#include <string>
#include <cmath>
#include <vector>
#include <iostream>

namespace py = pybind11;


int index(int a, int b)
{
    int ab = a>b ? a*(a+1)/2 + b : b*(b+1)/2 + a;
    return ab;
}

py::array_t<double> make_J(py::array_t<double> g, py::array_t<double> D)
{
    int i, j, k, l, ij, kl, ijkl;
    py::buffer_info g_info = g.request();    
    py::buffer_info D_info = D.request();    

    const double * g_data = static_cast<double *>(g_info.ptr);
    const double * D_data = static_cast<double *>(D_info.ptr);

    size_t nbf = g_info.shape[0];

    double eri_size = ((nbf * (nbf+1) ) * 0.5 * ((nbf * (nbf+1) ) * 0.5 + 1)) * 0.5;

    std::vector<double> eri(eri_size);

    // Access nbf^3 * i + nbf^2 * j + nbf * k + l
    for(size_t i = 0; i < nbf; i++)
    {
        for(size_t j = 0; j <= i; j++)
        {
            ij = index(i,j);
            for(size_t k = 0; k < nbf; k++)
            {
                for(size_t l = 0; l <= k; l++)
                {
                    kl = index(k,l);
                    ijkl = index(ij,kl); 
                    eri[ijkl] = g_data[nbf*nbf*nbf*i + nbf*nbf*j + nbf*k + l];
                }
            }
        }
    }

    std::vector<double> J_data(nbf * nbf);
//    std::vector<double> temp(nbf(nbf+1)/2);
    std::vector<double> temp(nbf*nbf);

    for(size_t i = 0; i < nbf; i++)
    {
        for(size_t j = 0; j <= i; j++)
        {
            std::fill(temp.begin(), temp.end(), 0);
            ij = index(i,j);
//            temp.setZero();
            for(size_t k = 0; k < nbf; k++)
            {
                for(size_t l = 0; l < nbf; l++)
                {
                    kl = index(k,l);
                    ijkl = index(ij,kl);
                    temp[k*nbf + l] = eri[ijkl];
                }
            }
            J_data[i * nbf + j] = LAWrap::dot(nbf*nbf, temp.data(), 1, D_data, 1);
            J_data[j * nbf + i] = J_data [ i * nbf + j];
        }
    }

    py::buffer_info Jbuff = 
        {
            J_data.data(),
            sizeof(double),
            py::format_descriptor<double>::format(),
            2,
            { nbf, nbf },
            { nbf * sizeof(double), sizeof(double) }
        };

//    std::cout << "c++ J starts" << std::endl;
//    for(auto i:J_data)
//        std::cout << i << std::endl;
//    std::cout << "c++ J ends" << std::endl;

    return py::array_t<double>(Jbuff);
} 


PYBIND11_PLUGIN(jk_mod)
{
    py::module jk("jk_mod", "JK module");

    jk.def("make_J", &make_J, "God I hope this makes J");

return jk.ptr();
}
