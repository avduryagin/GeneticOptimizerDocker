#pragma once
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "Python.h"
#include "main.h"
namespace py = pybind11;
using f_array = py::array_t<float, py::array::c_style | py::array::forcecast>;
using int_array = py::array_t<int, py::array::c_style | py::array::forcecast>;

class optimizer
{
	Population* optimizer_ptr=nullptr;
protected:
	f_array* data_ptr = nullptr;
	float* x=nullptr;
	float* y=nullptr;
	size_t npopul = 1, cast_number = 3, mutate_cell = 1, length = 1,allow_count=5,njobs=1;
	float bound = 0, threshold = 0.7f, epsilon = 1e-3f, tolerance = 0.7f, inh_threshold = 0.1f;
	bool mutate_random = false;

	void init(f_array*, int_array&, unsigned int, float, float, float, float, float, unsigned int, unsigned int, unsigned int, bool);


public:

	optimizer();
	~optimizer();
	void optimize(f_array* , int_array& , unsigned int , float , float ,float, float , float,  unsigned int , unsigned int ,unsigned int,bool);
	float val = 0;
	float rest = 0;
	size_t niter = 0;

};
class multijobs_optimizer:public optimizer
{private:
	PopulationParallel* optimizer_ptr = nullptr;
	size_t njobs=2;
protected:
	void init(f_array*, int_array&, unsigned int, float, float, float, float, float, unsigned int, unsigned int, unsigned int, bool, unsigned int);
public:
	using optimizer :: optimizer;
	multijobs_optimizer();
	~multijobs_optimizer();
	void optimize(f_array*, int_array&, unsigned int, float, float, float, float, float, unsigned int, unsigned int, unsigned int, bool, unsigned int);


};