#include "main.h"



Individ::Individ(size_t size,float bound,std::random_device& rd)
{
	this->size = size;
	this->bound = std::abs(bound);
	this->rest_ = 0;
	this->val_ = 0;
	this->code_ = new bool[this->size];
	std::fill_n(this->code_, this->size, false);	
	this->randint_ = std::uniform_int<size_t>(0, this->size - 1);
	this->random_generator_.seed(rd());
	

}
void Individ::pint_to_log(std::string& log,size_t index)
{
	std::string row;
	row = std::to_string(index)+". val:" + std::to_string(this->get_val()) + " rest:" + std::to_string(this->get_rest());
	row += "protected: " + std::to_string(this->isprotected()) + "inherited: " + std::to_string(this->isinherited());
	if (this->code_ == nullptr) { row += "code : nullptr"; }
	else{ row += "code :!nullptr"; }
	log.append(row+"\n");

}


Individ* Individ::copy(std::random_device& rd)
{
	Individ* c = new Individ(this->size, this->bound, rd);
	size_t i = 0;
	while (i<this->size)
	{
		c->code_[i] = this->code_[i];
		++ i;
	}
	c->val_=this->val_;
	c->rest_ = this->rest_;
	c->bound = this->bound;
	return c;
};
Individ::Individ(){}
Individ::~Individ()
{
	if (!(this->protected_)&&(this->npointer == 0)&&(this->code_!=nullptr)) 
	{
		delete[] this->code_;
		this->code_ = nullptr;
	}
}
void Individ::inherit(const Individ* mparent, const Individ* fparent)
{
	size_t index = this->randint();
	size_t i = 0;
	while(i<index)
	{
		this->code_[i] = mparent->code_[i];
		++i;
	}
	i = index;
	while (i < this->size)
	{
		this->code_[i] = fparent->code_[i];
		++i;
	}
	

};
const float Individ::get_rest() const
{
	return this->rest_;
};
const float Individ::get_val() const
{
	return this->val_;
};
void Individ::fit(float* x,size_t xlength,float* y,size_t ylength)
{
	if (!this->isvalid(xlength, ylength)) { return; }
	
	std::vector<size_t> index;	
	size_t i = 0,j=0;
	while(i<this->size)
	{
		index.push_back(i);
		++i;
	}
	this->shuffle(index);
	//std::shuffle(index.begin(), index.end(),this->random_generator_);
	i = 0;
	float s = 0;
	while(i<this->size)
	{
		j = index[i];
		s += std::abs(x[j]);
		if (s > this->bound) { break; }
		this->code_[j] = true;
		++i;
	}
	return;
}
const float Individ::value(float* x, size_t xlength, float* y, size_t ylength)
{
	if (!this->isvalid(xlength, ylength)) { return 0.; }
	size_t i= 0;
	this->val_ = 0;
	this->rest_ = 0;
	float v = 0;
	bool inverse = false;
	while (i<this->size)
	{
		this->rest_ += this->code_[i] * x[i];
		inverse = !this->code_[i];
		v += inverse * y[i];
		++i;
	}
	this->val_ = v + this->rest_;
	return this->val_;

};

const bool Individ::isvalid(size_t xlength, size_t ylength) const
{
	if ((xlength == ylength) && (xlength == this->size)) { return true; }
	else return false;
}
void Individ::mutate(size_t ncell) 
{
	size_t i = 0,j=0;
	
	while(i<ncell)
	{
		j = this->randint_(this->random_generator_);
		this->code_[j] = !this->code_[j];
		++i;
	}
};
void Individ::increase_npointer() { ++this->npointer; };
void Individ::decrease_npointer() 
{	
	if (this->npointer>0){ --this->npointer; }		
}
void Individ::erase_npointer() { this->npointer=0; }
const size_t Individ::randint()
{
	return this->randint_(this->random_generator_);
}
void Individ::shuffle(std::vector<size_t> & index)
{
	std::shuffle(index.begin(), index.end(), this->random_generator_);
}
const bool * Individ::code() const
{
	return this->code_;
}
void  Individ::delete_(bool force=false)
{
	if (this->code_ == nullptr) { return; }
	if (force) { this->set_protection(false); }
	if (this->isprotected()) { return;}
	this->set_inheritance(false);
	this->erase_npointer();
	delete this;
};
const bool Individ::isinherited() const { return this->inherited; };
void Individ::set_inheritance(const bool x) { this->inherited = x; };
void Individ::set_protection(const bool x) { this->protected_ = x; };
const bool Individ::isprotected() const { return this->protected_; };

Population::Population(size_t npopul_, float bound_, float* x_, float* y_, size_t length_,  float threshold_ = 0.7f, float inh_threshold_ = 0.1f, float epsilon_ = 1e-3f, float tolerance_ = 0.7f, size_t mutate_cell_ = 1, size_t cast_number_ = 3,bool random_mutate_=false)
{
	std::random_device rd;
	this->random_device=&rd;
	this->random_ = uniform_(0.f,1.f);
	this->random_generator_.seed(rd());


	this->npopul = npopul_;	
	this->bound = bound_;
	this->x = x_;
	this->y = y_;
	this->length = length_;
	this->threshold = threshold_;
	this->inh_threshold = inh_threshold_;
	this->epsilon = epsilon_;
	this->mutate_cell = mutate_cell_;
	this->tolerance = tolerance_;
	this->cast_number = cast_number_;
	this->random_mutate = random_mutate_;
	this->set_mutator(this->mutate_cell, this->random_mutate);

	
	this->path= "C:\\Users\\avduryagin\\source\\repos\\genetic_optimizer\\log_file.txt";
	//std::ofstream out;	
	//out.open(this->path);
	//this->out = &out;
	this->fit();

};

Population::~Population() 
{
	//this->out->close();
};
void Population::set_limits()
{
	this->minval = this->inf;
	this->maxval = -this->inf;
};

void Population::set_mutator(size_t ncell,bool random_mutate)
{
	if (this->mutator_ != nullptr) { delete this->mutator_; }
	
	if (random_mutate && (this->mutate_cell > 1))
	{
		this->mutator_ = new RandomMutator(this->mutate_cell, *this->random_device);
	}
	else
	{
		if (this->mutate_cell > 0) 
		{
			this->mutator_ = new Mutator(this->mutate_cell);
		}
		else{ this->mutator_ = new Mutator(1); }
	}
};
void Population::fit()
{
	size_t i = 0;
	float val = 0;
	this->set_limits();

	while (i<this->npopul)
	{
		Individ* individ = new Individ(this->length, this->bound, *this->random_device);	
		
		individ->fit(this->x, this->length, this->y, this->length);
		val = individ->value(this->x, this->length, this->y, this->length);
		if (val<this->minval)
		{
			this->minval = val;
			this->individ_index = i;
		}
		if (val > this->maxval) { this->maxval = val; }
		
		this->add_individ(i, individ,&this->population_);
		//Individ current = *this->population_[i];
		++i;
	}
	this->log.append(std::to_string(this->minval) + ",");
	this->optimal_individ = this->population_[this->individ_index];
	this->optimal_individ->set_protection(true);
	//(*this->out) << this->log;
}
void Population::add_individ(size_t index,Individ* individ_,dictionary_* population) 
{
	individ_->increase_npointer();
	(*population)[index] = individ_;
};
void Population::remove_individ(size_t index, Individ* individ_, dictionary_* population)
{
	individ_->decrease_npointer();
};

Individ* Population::cross(Individ* mparent, Individ* fparent) 
{
	Individ* child =new Individ(this->length, this->bound, *this->random_device);
	size_t mutate = 0;
	child->inherit(mparent, fparent);
	mutate = this->mutator_->randint();
	child->mutate(mutate);
	float val = child->value(this->x, this->length, this->y, this->length);
	if (std::abs(child->get_rest())>this->bound)
	{
		delete child;
		return nullptr;
	}
	return child;

};

void Population::selection(vector_* indices)
{
	std::unordered_map<size_t, Individ*>::iterator iter = this->population_.begin();
	size_t index = 0;
	float delta = 0;
	delta = (this->maxval - this->minval) * this->threshold+this->minval;
	while(iter!=this->population_.end())
	{
		iter->second->set_inheritance(false);
		index = iter->first;
		if (!(iter->second->get_val()>delta))
		{
			indices->push_back(index);
		}
		++iter;
	}
};
void Population::inherit(dictionary_* new_population, vector_* indices,size_t& ninherited, size_t& optimal_index,float& minval, float& maxval,size_t start=0)
{
	Individ* current = nullptr;
	size_t ninherited_ = 0;
	float delta = (this->maxval - this->minval) * this->inh_threshold + this->minval,val=0;
	vector_::iterator iter = indices->begin();
	while (iter!=indices->end())
	{
		current = this->population_[*iter];		
		val = current->get_val();
		if (val < delta)
		{
			current->decrease_npointer();
			current->set_inheritance(true);
			this->add_individ(start+ninherited_, current, new_population);
			
			if (val < minval) { minval = val; optimal_index = ninherited_; }
			if (val > maxval) { maxval = val; }
			++ninherited_;
		}
		++iter;
	}
	ninherited = ninherited_;
};
const size_t Population::cast(vector_* const indices) const 
{
	size_t length_ = indices->size();
	Individ generator = Individ(length_, 0., *this->random_device);
	size_t i = 0,j=0,index=0,optimal_index=0;
	float minval = this->inf,val=0;
	while(i<this->cast_number)
	{
		j = generator.randint();
		index = indices->at(j);
		val = this->population_.at(index)->get_val();
		if (val<minval)
		{
			minval = val;
			optimal_index = index;
		}
		++i;
	}
	return optimal_index;
};

void Population::get_new_population(dictionary_* new_population, size_t& optimal_index, float& minval, float& maxval, size_t start=0) 
{
	vector_ indices; 
	Individ* fparent=nullptr; Individ* mparent=nullptr;
	size_t mparent_index = 0, fparent_index = 0,optimal_index_=0;
	this->selection(&indices);	

	if (indices.size() == 0) {	return;	}	

	float minval_ = this->inf,maxval_=-this->inf,val=0;
	size_t i=0, k=0,index=0;	
	this->inherit(new_population, &indices, k,optimal_index_,minval_,maxval_,start);

	while((k<this->npopul)&&(i<this->maxiter))
	{
		fparent_index = this->cast(&indices);
		mparent_index = fparent_index;
		if (indices.size()>1)
		{
		while(mparent_index == fparent_index)
		{
			mparent_index = this->cast(&indices);
		}
		}


		
		++i;
		if (this->frandom()<this->tolerance)		{
			
			mparent = this->population_[mparent_index];
			fparent = this->population_[fparent_index];
			Individ* child = this->cross(mparent, fparent);
			if (child!=nullptr)
			{
				index = start + k;
				this->add_individ(index, child, new_population);
				val = child->get_val();
				if (val<minval_)
				{
					minval_ = val;
					optimal_index_ = index;

				}
				if (val > maxval_) { maxval_ = val; }
				++k;
			}
		}


	}
	maxval = maxval_;
	minval = minval_;
	optimal_index = optimal_index_;
};
const float Population::frandom() 
{
	return this->random_(this->random_generator_);
};

void Population::update_population(dictionary_* new_population, size_t optimal_index, float minval, float maxval) 
{
	if (new_population == nullptr) { return; }
	this->erase_pointers(&this->population_);
	this->population_ = *new_population;
	this->minval = minval;
	this->maxval = maxval;
	this->individ_index = optimal_index;
	//this->optimal_individ = this->population_[this->individ_index];

};
void Population::erase_pointers(dictionary_* population)
{
	if (population == nullptr) { return; }
	dictionary_::iterator iter = population->begin();	
	while(iter!=population->end())
	{
		if (iter->second->isinherited()||iter->second->isprotected()) { ++iter; continue; }
		iter->second->delete_(true);
		++iter;
	}
};
void Population::print_population(dictionary_* population)
{
	if (population == nullptr) { return; }
	dictionary_::iterator iter = population->begin();
	size_t i = 0;
	while (iter != population->end())
	{
		
		iter->second->pint_to_log(this->log,i);
		++iter;
		++i;
	}
};
void Population::optimize(const size_t allow_count) 
{
	this->niter = 0;
	size_t count = 0;
	float local_val=this->inf;
	Individ* local_individ = nullptr;
	Individ* solution = nullptr;
	float main_val = this->minval;
	std::string path = "C:\\Users\\avduryagin\\source\\repos\\genetic_optimizer\\log_file_";
	std::string path_ = "";
	std::string extension = ".txt";
	std::ofstream out;
	path_ = path + std::to_string(this->niter) + extension;
	out.open(path_);
	//this->out = &out;
	try {

		while (this->niter < this->epoch)
		{
			count = 0;
			local_val = this->inf;
			local_individ = nullptr;


			while (count < allow_count)
			{
				try {
					dictionary_ population;
					size_t optimal_index;
					float maxval_ = -this->inf;
					float minval_ = this->inf;
					this->get_new_population(&population, optimal_index, minval_, maxval_);					
					this->update_population(&population, optimal_index, minval_, maxval_);
					this->log.append(std::to_string(this->minval) + ",");


					if ((minval_ < local_val) && (std::abs(minval_ - local_val) > this->epsilon))
					{
						local_val = minval_;
						if (local_individ != nullptr) { local_individ->set_protection(false); }
						local_individ = this->population_[this->individ_index]->copy(*this->random_device);
						local_individ->set_protection(true);
					}
					++count;
					++this->niter;
					//path_ = path + std::to_string(this->niter) + extension;
					//out.open(path_);
				}
				catch (std::exception ex) 
				{ 
					log.append(ex.what());
					out << log;
					out.close();
					this->niter = this->epoch;
					break;
				}
			}
			if ((local_val < main_val) && (std::abs(local_val - main_val) > this->epsilon))
			{
				main_val = local_val;
				solution = local_individ;
				
			}
			else if (this->mutate_cell > 0)
			{
				--this->mutate_cell;
				this->set_mutator(this->mutate_cell, this->random_mutate);

			}

			else break;



		}
		if (solution != nullptr) { this->optimal_individ = solution; }
		//this->log.append("solution val " + std::to_string(this->optimal_individ->get_val()));
		out << this->log;
	}
	catch (...) { this->optimal_individ = nullptr; this->log.append("interupted by an error "); out << this->log; }

};
const size_t Population::niter_() const { return this->niter; };


Mutator::Mutator() {};
Mutator::~Mutator() {};
Mutator::Mutator(size_t ncell_)
{
	this->ncell = ncell_;
};

const size_t  Mutator::randint() { return this->ncell; }


RandomMutator::RandomMutator(size_t ncell,std::random_device& rd):Mutator{ncell}
{
	
	this->ncell = ncell;	
	this->randint_ = uniform(1, this->ncell);
	this->random_generator_.seed(rd());

};
RandomMutator::~RandomMutator() {};
const size_t RandomMutator::randint()
{
	return this->randint_(this->random_generator_);
};

PopulationParallel::PopulationParallel(size_t npopul_, float bound_, float* x_, float* y_, size_t length_, float threshold_ = 0.7f, float inh_threshold_ = 0.1f, float epsilon_ = 1e-3f, float tolerance_ = 0.7f, size_t mutate_cell_ = 1, size_t cast_number_ = 3, bool random_mutate_ = false,size_t njobs_=1)
	:Population{npopul_,bound_,  x_, y_, length_, threshold_, inh_threshold_,epsilon_ ,tolerance_ , mutate_cell_ ,  cast_number_ ,  random_mutate_ }
{
	std::thread th;	
	this->njobs = std::min(unsigned int(njobs_), th.hardware_concurrency()-1);
	std::mutex mut;
	this->mutex_ = &mut;
	
};

PopulationParallel::~PopulationParallel() {} //Population::~Population(); }

void PopulationParallel::get_new_population(dictionary_* new_population, size_t& optimal_index, float& minval, float& maxval,size_t start=0)
{
	//size_t njobs = 0;
	//std::thread th;

	
	
	
	if (this->njobs == 1) { Population::get_new_population(new_population, optimal_index, minval, maxval,start); return; }
	std::unordered_map<size_t, dictionary_> Populations;
	dictionary_ inh_popul;
	vector_ indices;	
	this->inheritance(&inh_popul, indices);


	std::unordered_map<size_t, std::thread> threads;
	//std::mutex m;
	size_t i = 0,ninh=0;
	ninh = inh_popul.size();
	while (i < this->njobs)
	{
		dictionary_ popul;
		Populations[i] = popul;
		//threads[i] = std::thread(&PopulationParallel::run, this, std::ref(Populations[i]), std::ref(indices), ninh, i * this->npopul);
		++i;

	}
	i = 0;
	
	while (i<this->njobs)
	{
		//dictionary_ popul;
		//Populations[i] = popul;
		threads[i]=std::thread(&PopulationParallel::run, this,std::ref(Populations[i]),std::ref(indices),ninh,i*this->npopul);
		++i;
		
	}
	if (inh_popul.size() > 0)
	{
		Populations[i] = inh_popul;
	}





	std::unordered_map<size_t, std::thread>::iterator iter = threads.begin();
	while (iter != threads.end())
	{		
		iter->second.join();
		++iter;
	}
	this->get_weight(Populations,*new_population,optimal_index,minval,maxval);

	//if (new_population->size() > 0) { std::cout << "size=" << new_population->size() << "\n"; }


}
void PopulationParallel::run(dictionary_ &population,vector_& indices, const size_t ninherited,size_t start=0)
{

	size_t optimal_index;
	float maxval_ = -this->inf;
	float minval_ = this->inf;
	this->breed(&population, indices, start, ninherited);
	//Population::get_new_population(&population, optimal_index, minval_, maxval_,start);


};
bool PopulationParallel::compare(pair A, pair B)
{
	if (A.second.second < B.second.second) { return true; }
	else return false;
}
void PopulationParallel::get_weight(std::unordered_map<size_t, dictionary_>& populations_,dictionary_& new_population, size_t& optimal_index, float& minval, float& maxval)
{
	//std::map<float, std::pair<size_t,size_t>> ordered_map;

	/*std::string path = "C:\\Users\\avduryagin\\source\\repos\\genetic_optimizer\\file_";
	std::string path_ = "";
	std::string extension = ".txt";
	std::ofstream out;
	path_ = path + std::to_string(niter) + extension;
	out.open(path_);*/
	vector_p vector_;

	std::unordered_map<size_t, dictionary_>::iterator iter = populations_.begin();
	dictionary_::iterator loc_iter;
	
	while(iter!=populations_.end())
	{
		loc_iter = iter->second.begin();
		while(loc_iter!=iter->second.end())
		{
			vector_.push_back(pair(iter->first, pair_(loc_iter->first, loc_iter->second->get_val())));	
			loc_iter->second->decrease_npointer();
			++loc_iter;
		}
		++iter;
	}
	std::sort(vector_.begin(), vector_.end(),this->compare);


	vector_p::iterator miter = vector_.begin();
	size_t i = 0,j=0,k=0;
	float val = 0;
	Individ individ_;
	while(miter!= vector_.end())
	{
		if (i < this->npopul) 
		{
			//new_population[i] = populations_[miter->first][miter->second.first];
			//new_population[i]->increase_npointer();
			this->add_individ(i, populations_[miter->first][miter->second.first], &new_population);
			val = miter->second.second;
			if (maxval < val) { maxval = val; }
		}
		else
		{

			populations_[miter->first][miter->second.first]->delete_();

		}
		++miter;
		++i;
		
	
	}
	optimal_index = 0;
	minval = new_population[optimal_index]->get_val();



};

void PopulationParallel::inheritance(dictionary_* new_population, vector_& indices)
{
	//vector_ indices;

	Population::selection(&indices);

	if (indices.size() == 0) { return; }

	float minval_ = this->inf, maxval_ = -this->inf;
	size_t i = 0, k = 0, optimal_index_ = 0;
	Population::inherit(new_population, &indices, k, optimal_index_, minval_, maxval_, 0);
}
void PopulationParallel::breed(dictionary_* new_population,vector_& indices,size_t start,size_t k)
{
	Individ* fparent = nullptr; Individ* mparent = nullptr;
	size_t mparent_index = 0, fparent_index = 0,i=0,index = 0;
	float val = 0;
	while ((k < this->npopul) && (i < this->maxiter))
	{
		fparent_index = this->cast(&indices);
		mparent_index = fparent_index;
		if (indices.size() > 1)
		{
			while (mparent_index == fparent_index)
			{
				mparent_index = this->cast(&indices);
			}
		}
		++i;
		if (this->frandom() < this->tolerance) {

			mparent = this->population_[mparent_index];
			fparent = this->population_[fparent_index];
			Individ* child = this->cross(mparent, fparent);
			if (child != nullptr)
			{
				index = start + k;
				//this->mutex_->lock();
				this->add_individ(index, child, new_population);
				//this->mutex_->unlock();
				val = child->get_val();
				++k;
			}
		}


	}
};