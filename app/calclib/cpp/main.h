#pragma once
#include <random>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include<string>
#include<memory>
#include<exception>
//#include "crtdbg.h"
/*
#ifdef  _CRTDBG_MAP_ALLOC

#define malloc(s)      _malloc_dbg(s,_NORMAL_BLOCK,__FILE__,__LINE__)
#define calloc(c,s)    _calloc_dbg(c,s,_NORMAL_BLOCK,__FILE__,__LINE__)
#define realloc(p,s)   _realloc_dbg(p,s,_NORMAL_BLOCK,__FILE__,__LINE__)
#define _expand(p,s)   _expand_dbg(p,s,_NORMAL_BLOCK,__FILE__,__LINE__)
#define free(p)        _free_dbg(p,_NORMAL_BLOCK)
#define _msize(p)      _msize_dbg(p,_NORMAL_BLOCK)

#endif  /* _CRTDBG_MAP_ALLOC */
/*
#ifdef _DEBUG
#ifdef _CRTDBG_MAP_ALLOC
#define new new(_NORMAL_BLOCK, __FILE__, __LINE__)
#endif /* _CRTDBG_MAP_ALLOC */
//#endif /* _DEBUG */


class Mutator
{
protected:
	size_t ncell = 0;
public:
	Mutator(size_t);
	Mutator();
	~Mutator();
	virtual const size_t randint();


};
class RandomMutator :public Mutator
{
	size_t ncell = 0;
	//typedef std::uniform_int_distribution<size_t> uniform;
	
	using uniform = std::uniform_int_distribution<size_t>;
	std::mt19937 random_generator_;
	uniform randint_;

public:
	using Mutator::Mutator;
	RandomMutator(size_t,std::random_device&);
	~RandomMutator();
	const size_t randint() override;

};
using boolean_ = std::unique_ptr<bool[]>;


class Individ
{
	
	float val_ = 0;
	float rest_ = 0;
	float bound = 0;
	boolean_ code_ = nullptr;
	size_t size = 1;	
	std::mt19937 random_generator_;
	std::uniform_int_distribution<size_t> randint_;
	const bool isvalid(size_t,size_t) const;
	

public:
	Individ(size_t, float,std::random_device &);
	Individ();
	~Individ();
	void fit(float*,size_t, float*,size_t);
	const float value(float*, size_t, float*, size_t);
	const float get_rest() const;
	const float get_val() const;	
	void pint_to_log(std::string& ,size_t index);
	void mutate(size_t);
	const size_t randint();
	void shuffle(std::vector<size_t> &);	
	bool at(size_t i) const;
	void inherit(const Individ*, const Individ*);
	Individ* copy(std::random_device&);
};

using individ_ptr = std::shared_ptr<Individ>;
using individ_ptru = std::unique_ptr<Individ>;

class Population
{
protected:
	using dictionary_ = std::unordered_map<size_t, individ_ptr>;
	using dictionary_iterator = std::unordered_map<size_t, individ_ptr>::iterator;
	std::random_device* random_device=nullptr;
	Mutator* mutator_ = nullptr;
	
	using vector_ = std::vector<size_t>;
	using uniform_ = std::uniform_real_distribution<float>;
	uniform_ random_;
	std::mt19937 random_generator_;
	dictionary_ population_;
	
	float* x = nullptr;
	float* y = nullptr;
	
	float bound = 0.f;
	float epsilon = 1e-3f;
	float tolerance = 0.7f;
	float threshold = 0.7f;
	float inh_threshold = 0.1f;
	const float inf= std::numeric_limits<float>::infinity();
	float minval = this->inf;
	float maxval = -this->inf;
	bool random_mutate = false;
	size_t npopul = 100;
	size_t maxiter = 100000;
	size_t length = 1;
	size_t mutate_cell = 1;
	size_t cast_number;
	size_t epoch = 1000;
	size_t niter = 0;	
	size_t individ_index = 0;
	std::string log;
	std::string path; 
	std::ofstream* out;
	void fit();
	void set_limits();
	void add_individ(size_t,individ_ptr,dictionary_*);	
	individ_ptr cross(individ_ptr, individ_ptr);
	void selection(vector_*);
	const size_t cast(const vector_&, individ_ptr) const;
	
	void inherit(dictionary_*, vector_*,size_t&, size_t&, float&, float&,size_t);
	const float frandom();
	void set_mutator(size_t,bool);	
	void print_population(dictionary_*);
	void update_population(dictionary_*, size_t, float, float);
	virtual void get_new_population(dictionary_*, size_t&, float&, float&,size_t);




	

public:
	individ_ptr optimal_individ=nullptr;
	Population(size_t, float, float*, float*, size_t, float, float,  float,float, size_t, size_t,bool);
	~Population();
	const size_t niter_() const;
	virtual void optimize(const size_t);
	
};

#include <thread>
#include <mutex>
#include <math.h>
#include <chrono>
#include <map>
class PopulationParallel:public Population
{
	using populations = std::unordered_map<size_t, dictionary_*>;
	populations populations_;
	using pair_ = std::pair<size_t, float>;
	using pair = std::pair<size_t,pair_>;
	using vector_p = std::vector<pair>;
	
	size_t njobs = 1;
	virtual void get_new_population(dictionary_*, size_t&, float&, float&,size_t) override;
	void run(dictionary_&,vector_&,const size_t,size_t);
	void get_weight(std::unordered_map<size_t, dictionary_>&,dictionary_&,size_t&,float&,float&);
	static bool compare(pair, pair);
	void inheritance(dictionary_*,vector_&);
	void breed(dictionary_*, vector_& , individ_ptr ,const size_t , size_t );	
	void sleep(int duration) {
		// simulate expensive operation
		std::this_thread::sleep_for(std::chrono::seconds(duration));
	}

public:
	using Population::Population;
	
	PopulationParallel(size_t, float, float*, float*, size_t, float, float, float, float, size_t, size_t, bool, size_t);
	~PopulationParallel();
};

