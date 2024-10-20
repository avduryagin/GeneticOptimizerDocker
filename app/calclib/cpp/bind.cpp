#include "bind.h"

optimizer::optimizer() {};
optimizer::~optimizer()
{
	if (this->x != nullptr) { delete[] this->x; }
	if (this->y != nullptr) { delete[] this->y; }
};

void optimizer::init(f_array* data_ptr_, int_array& indices, unsigned int npopul_, float bound_, float threshold_ = 0.7f, float inh_threshold_ = 0.1f, float epsilon_ = 1e-3f, float tolerance_ = 0.7f, unsigned int mutate_cell_ = 1, unsigned int cast_number_ = 3, unsigned int allow_count_ = 5, bool mutate_random_ = false) 
{
	this->data_ptr = data_ptr_;
	if (this->data_ptr->shape(1) != 2) { return; }
	this->length = this->data_ptr->shape(0);
	this->x = new float[this->length];
	this->y = new float[this->length];
	this->allow_count = size_t(allow_count_);
	this->npopul = size_t(npopul_);
	this->mutate_cell = size_t(mutate_cell_);
	this->bound = bound_;
	this->threshold = threshold_;
	this->inh_threshold = inh_threshold_;
	this->epsilon = epsilon_;
	this->tolerance = tolerance_;
	this->mutate_random = mutate_random_;

	size_t i = 0;
	while (i < this->length)
	{
		this->x[i] = data_ptr->at(i, 0);
		this->y[i] = data_ptr->at(i, 1);
		//log.append(std::to_string(this->x[i]) + " " + std::to_string(this->y[i]) + "\n");
		++i;
	}

};
void optimizer::optimize(f_array* data_ptr_, int_array& indices, unsigned int npopul_, float bound_, float threshold_ = 0.7f, float inh_threshold_ = 0.1f, float epsilon_ = 1e-3f, float tolerance_ = 0.7f, unsigned int mutate_cell_ = 1, unsigned int cast_number_ = 3, unsigned int allow_count_ = 5,bool mutate_random_=false)
{
	this->init(data_ptr_, indices, npopul_, bound_, threshold_, inh_threshold_, epsilon_, tolerance_, mutate_cell_, cast_number_, allow_count_, mutate_random_);
	
	std::string log;
	std::string path;
	std::ofstream out;
	path = "//app//log_file_.txt";	
	out.open(path);
	log.append("length: " + std::to_string(this->length)+"\n");
	log.append("npopul: " + std::to_string(this->npopul) + "\n");
	log.append("mutate_cell: " + std::to_string(this->mutate_cell) + "\n");
	log.append("allow_count: " + std::to_string(this->allow_count) + "\n");
	log.append("bound: " + std::to_string(this->bound) + "\n");
	log.append("threshold: " + std::to_string(this->threshold) + "\n");
	log.append("epsilon: " + std::to_string(this->epsilon) + "\n");
	log.append("tolerance: " + std::to_string(this->tolerance) + "\n");
	log.append("njobs: " + std::to_string(this->njobs) + "\n");


	Population optimizer_ = Population(this->npopul, this->bound, this->x, this->y, this->length, this->threshold, this->inh_threshold, this->epsilon, this->tolerance, this->mutate_cell, this->cast_number, this->mutate_random);
	optimizer_.optimize(this->allow_count);
	individ_ptr pointer = optimizer_.optimal_individ;
	Individ* individ = pointer.get();
	this->val = individ->get_val();
	this->rest = individ->get_rest();

	size_t i = 0;
	while (i < this->length)
	{
		indices.mutable_at(i) = int(individ->at(i));
		++i;
	}
	this->optimizer_ptr = &optimizer_;
	this->niter = optimizer_.niter_();
	out << log;


	
};
void multijobs_optimizer::init(f_array* data_ptr_, int_array& indices, unsigned int npopul_, float bound_, float threshold_ = 0.7f, float inh_threshold_ = 0.1f, float epsilon_ = 1e-3f, float tolerance_ = 0.7f, unsigned int mutate_cell_ = 1, unsigned int cast_number_ = 3, unsigned int allow_count_ = 5, bool mutate_random_ = false, unsigned int njobs_ = 1)
{
	optimizer::init(data_ptr_, indices, npopul_, bound_, threshold_, inh_threshold_, epsilon_, tolerance_, mutate_cell_, cast_number_, allow_count_, mutate_random_);
	this->njobs = njobs_;
};
multijobs_optimizer::multijobs_optimizer() :optimizer{} {};
multijobs_optimizer::~multijobs_optimizer() {};
void multijobs_optimizer::optimize(f_array* data_ptr_, int_array& indices, unsigned int npopul_, float bound_, float threshold_ = 0.7f, float inh_threshold_ = 0.1f, float epsilon_ = 1e-3f, float tolerance_ = 0.7f, unsigned int mutate_cell_ = 1, unsigned int cast_number_ = 3, unsigned int allow_count_ = 5, bool mutate_random_ = false, unsigned int njobs_ = 1)
{
	this->init(data_ptr_, indices, npopul_, bound_, threshold_, inh_threshold_, epsilon_, tolerance_, mutate_cell_, cast_number_, allow_count_, mutate_random_,njobs_);
	std::string log;
	std::string path;
	std::ofstream out;
	path = "//app//log_file_.txt";	
	out.open(path);
	log.append("length: " + std::to_string(this->length) + "\n");
	log.append("npopul: " + std::to_string(this->npopul) + "\n");
	log.append("mutate_cell: " + std::to_string(this->mutate_cell) + "\n");
	log.append("allow_count: " + std::to_string(this->allow_count) + "\n");
	log.append("bound: " + std::to_string(this->bound) + "\n");
	log.append("threshold: " + std::to_string(this->threshold) + "\n");
	log.append("epsilon: " + std::to_string(this->epsilon) + "\n");
	log.append("tolerance: " + std::to_string(this->tolerance) + "\n");
	log.append("njobs: " + std::to_string(this->njobs) + "\n");
	out << log;
	out.close();

	PopulationParallel optimizer_ = PopulationParallel(this->npopul, this->bound, this->x, this->y, this->length, this->threshold, this->inh_threshold, this->epsilon, this->tolerance, this->mutate_cell, this->cast_number, this->mutate_random, this->njobs);
	try {
		//path = "C:\\Users\\avduryagin\\source\\repos\\genetic_optimizer\\log_file_1.txt";
		//out.open(path);
		optimizer_.optimize(this->allow_count);
		//log.append("optimizer passed\n");
		//out << log;
		//out.close();
		//path = "C:\\Users\\avduryagin\\source\\repos\\genetic_optimizer\\log_file_2.txt";
		//out.open(path);
		;
		individ_ptr  pointer = optimizer_.optimal_individ;
		if (pointer == nullptr) { throw std::invalid_argument("Empty answer"); log.append("thrown an exception "); out << log; out.close(); }
		Individ* individ = pointer.get();
		//log.append("individ passed\n");
		//out << log;
		//out.close();

		this->val = individ->get_val();
		//log.append("val: " + std::to_string(this->val) + "\n");
		this->rest = individ->get_rest();
		//log.append("rest: " + std::to_string(this->rest) + "\n");
		this->niter = optimizer_.niter_();
		//log.append("niter: " + std::to_string(this->niter) + "\n");
		size_t i = 0;
		while (i < this->length)
		{
			indices.mutable_at(i) = int(individ->at(i));
			//log.append(std::to_string(indices.at(i)) + ",");
			++i;
		}
		//log.append("\n code passed: " + std::to_string(this->niter) + "\n");	

		//out << log;
		//out.close();
	}
	catch (...)
	{
		log.append("error");
		out << log;
		out.close();
	}
};