#include <shogun/base/init.h>
#include <shogun/lib/common.h>
#include <shogun/lib/SGVector.h>
#include <shogun/io/SGIO.h>
#include <shogun/io/CSVFile.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/statistics/KernelMeanMatching.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <iterator>
#include <vector>
#include <map>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <libconfig.h++>

using namespace shogun;
using namespace std;
using namespace libconfig;

Config config;
enum Norm
{
	L1,
	L2
};

CSparseFeatures<double>* trainFeatsX;
CSparseFeatures<double>* trainFeatsY;
vector<int> nodeI, nodeJ;
vector<double> edgeWt;
map<int,int> nodeToIdx_x, nodeToIdx_y;
int numFactors, d_x, d_y, numEdges, n_x, n_y;
double lambda, eta;
vector <double> betas; // betas[t]: weigh specific and general parts
vector <double> sigmas;
vector< vector <double> > U;		// numfeats x numfact
vector< vector <double> > gradU;	// numfeats x numfact
vector< vector <double> > V;		// numfeats x numfact
vector< vector <double> > gradV;	// numfeats x numfact
vector< vector <double> > XU;		// numNodes_X x numfact
vector< vector <double> > YV;		// numNodes_Y x numfact 
vector< vector <double> > XgradU;		// numNodes_X x numfact
vector< vector <double> > YgradV;		// numfact x numNodes_Y
vector<double> cost_vec; 	// numEdges
double cost_factor = 1; // to weigh the squared loss
Norm norm_param;

const double MIN_ETA = 0.0000001;
const double INNER_THRESH = 0.0000001;
const int MAX_INNER_ITER = 40;
const int MIN_INNER_ITER = 3;

// task level parameters
vector< vector< vector <double> > > XSY;		// numTasks x numNodes_X x numNodes_Y
vector< vector< vector <double> > > XgradSY;		// numTasks x numNodes_X x numNodes_Y
vector< vector< vector <double> > > S;		// numTasks x numfeats1 x numfeats2
vector< vector< vector <double> > > gradS;	// num_tasks x numfeats1 x numfeats2
vector<int> taskStartIndices, testTaskStartIndices;
int num_tasks;

void print_message(FILE* target, const char* str)
{
        fprintf(target, "%s", str);
}

void print_warning(FILE* target, const char* str)
{
        fprintf(target, "%s", str);
}

void print_error(FILE* target, const char* str)
{
        fprintf(target, "%s", str);
}

void set_zero_3d(vector< vector< vector <double> > > &arr, int m, int n, int o)
{
	#pragma omp parallel for 
	for(int row=0; row<m; row++) 
		for(int col=0; col<n; col++) 
			for(int col2=0; col2<o; col2++) 
				arr[row][col][col2]=0;
}

void set_zero(vector< vector <double> > &arr, int m, int n)
{
	#pragma omp parallel for 
	for(int row=0; row<m; row++) 
		for(int col=0; col<n; col++) 
			arr[row][col]=0;
}

void read_config_file(const char* filename)
{
    try
    {
	config.readFile(filename);
    }
    catch(FileIOException &fioex)
    {
	std::cerr << "I/O error while reading file." << std::endl;
    }
    catch(ParseException &pex)
    {
	std::cerr << "Parse error at line" << ":" << pex.getLine() << " - " << pex.getError() << std::endl;
    }
}

void copy_matrix(vector< vector <double> > &source, int nrows, int ncols, vector< vector <double> > &target)
{
    #pragma omp parallel for 
    for(int r=0; r<nrows; r++)
        for(int c=0; c<ncols; c++)
            target[r][c] = source[r][c];
}

void product_feature_factor(CSparseFeatures<double>* feats, vector< vector <double> > &factor, vector< vector <double> > &result) 
{
	SGSparseMatrix<double> spmat = feats->get_sparse_feature_matrix();
	// iterate over all nodes
	#pragma omp parallel for 
	for(int nIdx=0; nIdx<spmat.num_vectors; nIdx++)
	{
		for ( int d=0; d<spmat[nIdx].num_feat_entries; d++ )
		{
			int featIdx = spmat[nIdx].features[d].feat_index;
			double featVal = spmat[nIdx].features[d].entry;
			for( int k=0; k<numFactors; k++ )
				result[nIdx][k] += featVal*factor[featIdx][k];
		}
		// debug print
		//for( int k=0; k<numFactors; k++ )
		//	printf("XU[%d][%d]: %.10g\n",nIdx,k,XU[nIdx][k]);
	}
}

// computes XSY, XgradSY
void compute_triproduct(int task, vector< vector <double> > &factor, vector< vector <double> > &result)
{
	set_zero(result,n_x,n_y);
	#pragma omp parallel for
    for( int e=taskStartIndices[task]; e<taskStartIndices[task+1]; e++ )
	{
	  if(cost_vec[e]>0) 
	  {
		int nI = nodeToIdx_x[nodeI[e]];	
		int nJ = nodeToIdx_y[nodeJ[e]];	
		//printf("%d = %d,%d = %d,%d\n",e,nodeI[e],nodeJ[e],nI,nJ);
        SGSparseVector<double> x = trainFeatsX->get_sparse_feature_vector(nI);
        SGSparseVector<double> y = trainFeatsY->get_sparse_feature_vector(nJ);
		for ( int dx=0; dx<x.num_feat_entries; dx++ )
		{
			int xfeat = x.features[dx].feat_index;
			double xval = x.features[dx].entry;
			for ( int dy=0; dy<y.num_feat_entries; dy++ )
			{
				int yfeat = y.features[dy].feat_index;
				double yval = y.features[dy].entry;
				result[nI][nJ] += xval * factor[xfeat][yfeat] * yval;
			}
		}
	  }
	}
}

void compute_intermediate()
{
	product_feature_factor(trainFeatsX, U, XU);
	product_feature_factor(trainFeatsY, V, YV);
	for(int t=0; t<num_tasks; t++) {
		compute_triproduct(t,S[t],XSY[t]);
	}
}

double l1_norm(vector< vector <double> > &mat, int rows, int cols)
{
	double l1norm = 0;
	#pragma omp parallel for reduction(+:l1norm)
	for ( int i=0; i<rows; i++ )
		for( int j=0; j<cols; j++ )
			l1norm += abs(mat[i][j]);
	return l1norm;
}

double l2_norm(vector< vector <double> > &mat, int rows, int cols)
{
	double l2norm = 0;
	#pragma omp parallel for reduction(+:l2norm)
	for ( int i=0; i<rows; i++ )
		for( int j=0; j<cols; j++ )
			l2norm += pow(mat[i][j],2);
	return l2norm;
}


double fro_norm(vector< vector <double> > &mat, int rows, int cols)
{
	double frob = 0;
	for ( int i=0; i<rows; i++ )
		for( int j=0; j<cols; j++ )
			frob += pow(mat[i][j],2);
	return frob;
}

double compute_fn_value()
{
	double loss=0, zerone_err=0, allNormS=0;
	vector<double> taskLoss(num_tasks);
	// squared error
	//#pragma omp parallel for reduction(+:loss,zerone_err) -- commented as resulting in inconsistency
	for( int t=0; t<num_tasks; t++ )
	{
	  for( int e=taskStartIndices[t]; e<taskStartIndices[t+1]; e++ )
	  {
	    if(cost_vec[e]>0) 
	    {
		int nI = nodeToIdx_x[nodeI[e]];	
		int nJ = nodeToIdx_y[nodeJ[e]];	
		double predIJ=0;
		for( int k=0; k<numFactors; k++ )
			predIJ += XU[nI][k] * YV[nJ][k];
		double mtlpred = betas[t]*predIJ + (1-betas[t])*XSY[t][nI][nJ];
		double prevpred = predIJ + XSY[t][nI][nJ];
		taskLoss[t] += cost_vec[e] * pow(edgeWt[e] - mtlpred,2);
	        if((mtlpred>=0.5 && edgeWt[e]<1) || (mtlpred<0.5 && edgeWt[e]==1))
        	    zerone_err ++;
	    }
	  }
	  loss += taskLoss[t];
	  double normS = (norm_param==L1) ?  l1_norm(S[t], d_x, d_y) : l2_norm(S[t], d_x, d_y);
	  fprintf(stderr,"Task-%d  Loss: %.10g Norm-S: %.10g \n",t,taskLoss[t],normS);
	  allNormS += sigmas[t] * normS;
	}

	// regularizers
	double frobU = fro_norm(U, d_x, numFactors);
	double frobV = fro_norm(V, d_y, numFactors);

    fprintf(stderr,"\nLoss: %.10g \t 0/1 error: %.10g  Averaged: %.10g\n",loss,zerone_err,zerone_err/numEdges);

	loss = loss/cost_factor + lambda*frobU + lambda*frobV + allNormS;

	cerr << "Function: " << loss << "\tFrobU: " << frobU << "\tFrobV: " << frobV << "\tFrobS: " << allNormS << endl;
	return loss;
}

double elewise_prod(vector< vector <double> > &mat1, vector< vector <double> > &mat2, int rows, int cols)
{
	double prod = 0;
	#pragma omp parallel for reduction(+:prod)
	for ( int i=0; i<rows; i++ )
		for( int j=0; j<cols; j++ )
			prod += mat1[i][j] * mat2[i][j];
	return prod;
}

double get_step_size_U()
{
	double numer=0, denom=0;
	set_zero(XgradU,n_x,numFactors);
	product_feature_factor(trainFeatsX, gradU, XgradU);
	#pragma omp parallel for reduction(+:numer,denom)
	for( int t=0; t<num_tasks; t++ )
	{
	  for( int e=taskStartIndices[t]; e<taskStartIndices[t+1]; e++ )
	  {
	    if(cost_vec[e]>0) 
	    {
		int nI = nodeToIdx_x[nodeI[e]];	
		int nJ = nodeToIdx_y[nodeJ[e]];	
		double pred=0, gradPred=0;
		for( int k=0; k<numFactors; k++ ) {
			pred += XU[nI][k] * YV[nJ][k];
			gradPred += XgradU[nI][k] * YV[nJ][k];
		}
		gradPred = betas[t]*gradPred;
		double mtlpred = betas[t]*pred + (1-betas[t])*XSY[t][nI][nJ];
		double residual = (edgeWt[e] - mtlpred);
		numer += cost_vec[e] * residual * gradPred;
		denom += cost_vec[e] * gradPred*gradPred;
	    }
	  }
	}
	denom = denom/cost_factor + lambda * fro_norm(gradU, d_x, numFactors);
	numer = numer/cost_factor + lambda * elewise_prod(U, gradU, d_x, numFactors);
	fprintf(stderr,"_|^ Step size: %.10g\n",-(numer/denom));
	return -(numer/denom);
}

double get_step_size_V()
{
	double numer=0, denom=0;
	set_zero(YgradV,n_y,numFactors);
	product_feature_factor(trainFeatsY, gradV, YgradV);
	#pragma omp parallel for reduction(+:numer,denom)
	for( int t=0; t<num_tasks; t++ )
	{
	  for( int e=taskStartIndices[t]; e<taskStartIndices[t+1]; e++ )
	  {
	    if(cost_vec[e]>0) 
	    {
		int nI = nodeToIdx_x[nodeI[e]];	
		int nJ = nodeToIdx_y[nodeJ[e]];	
		double pred=0, gradPred=0;
		for( int k=0; k<numFactors; k++ ) {
			pred += XU[nI][k] * YV[nJ][k];
			gradPred += XU[nI][k] * YgradV[nJ][k];
		}
		gradPred = betas[t]*gradPred;
		double mtlpred = betas[t]*pred + (1-betas[t])*XSY[t][nI][nJ];
		double residual = (edgeWt[e] - mtlpred);
		numer += cost_vec[e] * residual * gradPred;
		denom += cost_vec[e] * gradPred*gradPred;
	    }
	  }
	}
	denom = denom/cost_factor + lambda * fro_norm(gradV, d_y, numFactors);
	numer = numer/cost_factor + lambda * elewise_prod(V, gradV, d_y, numFactors);
	//fprintf(stderr,"_|^ Step size: %.10g/%.10g = %.10g\n",numer,denom,-(numer/denom));
	fprintf(stderr,"_|^ Step size: %.10g\n",-(numer/denom));
	return -(numer/denom);
}

double get_step_size_S(int task, bool skipNorm)
{
	double numer=0, denom=0;
	set_zero(XgradSY[task],n_x,n_y);
	compute_triproduct(task,gradS[task],XgradSY[task]);
	#pragma omp parallel for reduction(+:numer,denom)
  	for( int e=taskStartIndices[task]; e<taskStartIndices[task+1]; e++ )
	{
	  if(cost_vec[e]>0) 
	  {
		int nI = nodeToIdx_x[nodeI[e]];	
		int nJ = nodeToIdx_y[nodeJ[e]];	
		double predIJ=0, gradPredIJ=0;
		for( int k=0; k<numFactors; k++ )
			predIJ += XU[nI][k] * YV[nJ][k];
		double mtlpred = betas[task]*predIJ + (1-betas[task])*XSY[task][nI][nJ];
		double residual = (mtlpred - edgeWt[e]);
		numer += cost_vec[e] * residual * (1-betas[task]) * XgradSY[task][nI][nJ];
		denom += cost_vec[e] * pow((1-betas[task]) * XgradSY[task][nI][nJ],2.0);
	  }
	}
	if(!skipNorm)
	{
		denom = denom/cost_factor + sigmas[task] * l2_norm(gradS[task], d_x, d_y);
		numer = numer/cost_factor + sigmas[task] * elewise_prod(S[task], gradS[task], d_x, d_y);
	}
	fprintf(stderr,"_|^ Step size: %.10g/%.10g = %.10g\n",numer,denom,(numer/denom));
	//fprintf(stderr,"_|^ Step size: %.10g\n",-(numer/denom));
	return (numer/denom);
}

/**** DEBUG : check gradient alright ****/
void check_gradient_computation(vector< vector <double> > &factor, vector< vector <double> > &grad, vector< vector <double> > &product, CSparseFeatures<double>* feats, int numFeats, int numExamples) 
{
	fprintf(stdout,"=============== GRADIENT CHECKING MACHINE ===============\n");
	double eps=0.0001;
	for ( int d=0; d<numFeats; d++ ) 
	{
		for( int k=0; k<numFactors; k++ ) 
		{
			double copy = factor[d][k];
			factor[d][k] += eps;
			set_zero(product,numExamples,numFactors);
			product_feature_factor(feats, factor, product);
			double plus = compute_fn_value();
			factor[d][k] = copy-eps;
			set_zero(product,numExamples,numFactors);
			product_feature_factor(feats, factor, product);
			double minus = compute_fn_value();
			factor[d][k] = copy;
			double fnDiff = (plus - minus)/(2*eps);
			fprintf(stdout,"[%d][%d] grad: %.10g fnChange: %.10g plus:%.10g minus:%.10g\n",d,k,grad[d][k],fnDiff,plus,minus);
		}
	}
	fprintf(stdout,"==============================\n");
}

void check_gradient_computation_S(int task) 
{
	fprintf(stdout,"=============== GRADIENT CHECKING MACHINE ===============\n");
	double eps=0.00000001;
	for ( int d=0; d<d_x; d++ ) 
	{
		for( int k=0; k<d_y; k++ ) 
		{
			double copy = S[task][d][k];
			S[task][d][k] = copy+eps;
			compute_triproduct(task,S[task],XSY[task]);
			double plus = compute_fn_value();
			S[task][d][k] = copy-eps;
			compute_triproduct(task,S[task],XSY[task]);
			double minus = compute_fn_value();
			double fnDiff = (plus - minus)/(2*eps);
			fprintf(stdout,"[%d][%d] grad: %.10g fnChange: %.10g plus:%.10g minus:%.10g\n",d,k,gradS[task][d][k],fnDiff,plus,minus);
			S[task][d][k] = copy;
		}
	}
	fprintf(stdout,"==============================\n");
}

void compute_gradU() 
{
}
/********************/

void least_squares_U()
{
    //compute_intermediate();
    bool converge = false;
    double fnVal=FLT_MAX, oldVal;
    int iters = 0, fnInc = 0;
    while(!converge) {
	// compute gradient w.r.t U
	set_zero(gradU,d_x,numFactors);
       #pragma omp parallel
       {
        vector< vector <double> > localgrad;
        for( int i = 0 ; i < d_x; i++ ) {
                vector<double> temp(numFactors);
                localgrad.push_back(temp);
        }

    #pragma omp for 
	for( int t=0; t<num_tasks; t++ )
	{
	  for( int e=taskStartIndices[t]; e<taskStartIndices[t+1]; e++ )
	  {
	    if(cost_vec[e]>0) {
		int nI = nodeToIdx_x[nodeI[e]];	
		int nJ = nodeToIdx_y[nodeJ[e]];	
		SGSparseVector<double> xI = trainFeatsX->get_sparse_feature_vector(nI);
		double predIJ=0, term2=0;
		for( int k=0; k<numFactors; k++ )
			predIJ += XU[nI][k] * YV[nJ][k];
		for( int k=0; k<numFactors; k++ )
		{
			double mtlpred = betas[t]*predIJ + (1-betas[t])*XSY[t][nI][nJ];
			term2 = cost_vec[e] * (mtlpred - edgeWt[e]) * YV[nJ][k];
			for ( int d=0; d<xI.num_feat_entries; d++ )
			{
				int featIdx = xI.features[d].feat_index;
				double featVal = xI.features[d].entry;
				//gradU[featIdx][k] += 2* term2 * featVal;
				localgrad[featIdx][k] += 2* term2 * featVal;
			}
		}
	    }
	  }
	} // end num tasks loop
	//#pragma omp barrier
	#pragma omp critical
	{
	    for( int i = 0 ; i < d_x; i++ )
		for( int k=0; k<numFactors; k++ )
			gradU[i][k] += localgrad[i][k] / cost_factor;
	}

	#pragma omp for
        for( int i = 0 ; i < d_x; i++ )
	{
           for( int k=0; k<numFactors; k++ )
		gradU[i][k] += 2* lambda*U[i][k];
	}
		localgrad.clear();
       } 
       // end parallel

	//check_gradient_computation(U, gradU, XU, trainFeatsX, d_x, n_x);
	// update U
	double maxGrad=0;
	double localeta = get_step_size_U();
	if(localeta < 0)
		localeta = eta;
	//#pragma omp parallel for 
	for ( int d=0; d<d_x; d++ )
		for( int k=0; k<numFactors; k++ ) {
			U[d][k] = U[d][k] - localeta * gradU[d][k];
			if(abs(gradU[d][k])>maxGrad)
				maxGrad = abs(gradU[d][k]);
		}
	// update variables dependent on U
	set_zero(XU,n_x,numFactors);
	product_feature_factor(trainFeatsX, U, XU);
	iters ++;
	// check convergence
	oldVal = fnVal;
        fnVal = compute_fn_value();
        double improvement = (oldVal-fnVal)/oldVal;
	fprintf(stderr,"Iteration: %d  Improvement: %.10g  Max. grad: %.10g\n",iters,improvement,maxGrad);
	if(improvement < -0.00001 ) {
                fnInc++;
        }

        if((abs(improvement) < INNER_THRESH && iters > MIN_INNER_ITER) || 
			iters > MAX_INNER_ITER || localeta < MIN_ETA)
                converge = true;
        if(fnInc>=1) {
                fprintf(stderr,"TOO MANY INCREASES !! I QUIT NOW :-(\n");
                return;
        }
    }
}

void least_squares_V()
{
    //compute_intermediate();
    bool converge = false;
    double fnVal=FLT_MAX, oldVal;
    int iters = 0, fnInc = 0;
    while(!converge) {	
	// compute gradient w.r.t U
	set_zero(gradV,d_y,numFactors);
       #pragma omp parallel
       {

        vector< vector <double> > localgrad;
        for( int i = 0 ; i < d_y; i++ ) {
                vector<double> temp(numFactors);
                localgrad.push_back(temp);
        }

    #pragma omp for
	for( int t=0; t<num_tasks; t++ )
	{
	  for( int e=taskStartIndices[t]; e<taskStartIndices[t+1]; e++ )
	  {
	    if(cost_vec[e]>0) {
		int nI = nodeToIdx_x[nodeI[e]];	
		int nJ = nodeToIdx_y[nodeJ[e]];	
		SGSparseVector<double> yJ = trainFeatsY->get_sparse_feature_vector(nJ);
		double predIJ=0, term2=0;
		for( int k=0; k<numFactors; k++ )
			predIJ += XU[nI][k] * YV[nJ][k];
		for( int k=0; k<numFactors; k++ )
		{
			double mtlpred = betas[t]*predIJ + (1-betas[t])*XSY[t][nI][nJ];
			term2 = cost_vec[e] * (mtlpred - edgeWt[e]) * XU[nI][k];
			for ( int d=0; d<yJ.num_feat_entries; d++ )
			{
				int featIdx = yJ.features[d].feat_index;
				double featVal = yJ.features[d].entry;
				localgrad[featIdx][k] += 2* term2 * featVal;
				//gradV[featIdx][k] += 2* term2 * featVal;
			}
		}
	    }
	  }
	} // end tasks loop
	#pragma omp critical
	{
	    for( int i = 0 ; i < d_y; i++ )
		for( int k=0; k<numFactors; k++ )
			gradV[i][k] += localgrad[i][k] / cost_factor;
	}

	#pragma omp for
	for( int i = 0 ; i < d_y; i++ )
	{
		for( int k=0; k<numFactors; k++ )
			gradV[i][k] += 2 * lambda*V[i][k];

	}
		localgrad.clear();
      }

	//check_gradient_computation(V, gradV, YV, trainFeatsY, d_y, n_y);
	// update V
	double maxGrad=0;
	double localeta = get_step_size_V();
	if(localeta < 0)
		localeta = eta;
	//#pragma omp parallel for 
	for ( int d=0; d<d_y; d++ )
		for( int k=0; k<numFactors; k++ ) {
			V[d][k] = V[d][k] - localeta * gradV[d][k];
			if(abs(gradV[d][k])>maxGrad)
				maxGrad = abs(gradV[d][k]);
		}
	// update variables dependent on V
	set_zero(YV,n_y,numFactors);
	product_feature_factor(trainFeatsY, V, YV);
	iters ++ ;
	// check convergence
	oldVal = fnVal;
        fnVal = compute_fn_value();
        double improvement = (oldVal-fnVal)/oldVal;
	fprintf(stderr,"Iteration: %d  Improvement: %.10g  Max. grad: %.10g\n",iters,improvement,maxGrad);
	if(improvement < -0.00001) {
                fnInc++;
        }

        if((abs(improvement) < INNER_THRESH && iters > MIN_INNER_ITER) || 
			iters > MAX_INNER_ITER || localeta < MIN_ETA )
                converge = true;

        if(fnInc>=1) {
                fprintf(stderr,"TOO MANY INCREASES !! I QUIT NOW :-(\n");
                return;
        }
    } 
}

void least_squares_S(int task)
{
	printf("[LeastSquares-S] Task: %d  start: %d  end: %d\n",task,taskStartIndices[task],taskStartIndices[task+1]);
    bool converge = false;
    double fnVal, oldVal;
	fnVal = compute_fn_value();
    int iters = 0, fnInc = 0;
    while(!converge) {	
	// compute gradient w.r.t S
	set_zero(gradS[task],d_x,d_y);
       #pragma omp parallel
       {
        vector< vector <double> > localgrad;
        for( int i = 0 ; i < d_x; i++ ) {
                vector<double> temp(d_y);
                localgrad.push_back(temp);
        }

     #pragma omp for
	for( int e=taskStartIndices[task]; e<taskStartIndices[task+1]; e++ )
	{
	    if(cost_vec[e]>0) {
		int nI = nodeToIdx_x[nodeI[e]];	
		int nJ = nodeToIdx_y[nodeJ[e]];	
		SGSparseVector<double> xI = trainFeatsX->get_sparse_feature_vector(nI);
		SGSparseVector<double> yJ = trainFeatsY->get_sparse_feature_vector(nJ);
		double predIJ=0, term2=0;
		for( int k=0; k<numFactors; k++ )
			predIJ += XU[nI][k] * YV[nJ][k];
		double mtlpred = betas[task]*predIJ + (1-betas[task])*XSY[task][nI][nJ];
		term2 =  2 * cost_vec[e] * (mtlpred - edgeWt[e]);
		for ( int dx=0; dx<xI.num_feat_entries; dx++ )
		{
			int xfeat = xI.features[dx].feat_index;
			double xval = xI.features[dx].entry;
			for ( int dy=0; dy<yJ.num_feat_entries; dy++ )
			{
				int yfeat = yJ.features[dy].feat_index;
				double yval = yJ.features[dy].entry;
				//gradS[task][xfeat][yfeat] += term2 * xval * yval;
				localgrad[xfeat][yfeat] += term2 * xval * yval;
			}
		}
	    }
	}
	#pragma omp critical
	{
	    for( int d1 = 0 ; d1 < d_x; d1++ )
		for( int d2 = 0; d2 < d_y; d2++ )
			gradS[task][d1][d2] += localgrad[d1][d2] / cost_factor;
	} 
	localgrad.clear();
  }

	//check_gradient_computation_S(task);
	// update S

	double localeta = (norm_param==L1) ? get_step_size_S(task,true) : get_step_size_S(task,false);
	if(localeta < 0)
		localeta = eta;
	double maxGrad=0;
	#pragma omp parallel for
	for( int d1 = 0 ; d1 < d_x; d1++ )
	{
		for( int d2 = 0; d2 < d_y; d2++ )
		{
			S[task][d1][d2] = S[task][d1][d2] - localeta * gradS[task][d1][d2];
			// soft thresholding for L1 norm
			if(norm_param == L1)
			{
				if(S[task][d1][d2] > sigmas[task])
					S[task][d1][d2] = S[task][d1][d2] - sigmas[task];
				else if(S[task][d1][d2] < -sigmas[task])
					S[task][d1][d2] = S[task][d1][d2] + sigmas[task];
				else
					S[task][d1][d2] = 0;

			}
			if(abs(gradS[task][d1][d2])>maxGrad)
				maxGrad = abs(gradS[task][d1][d2]);
		}
	}
	// update variables dependent on S
	compute_triproduct(task,S[task],XSY[task]);
	iters ++ ;
	// check convergence
	oldVal = fnVal;
        fnVal = compute_fn_value();
        double improvement = (oldVal-fnVal)/oldVal;
	fprintf(stderr,"Iteration: %d  Improvement: %.10g  Max. grad: %.10g\n",iters,improvement,maxGrad);
	if(improvement < -0.00001) {
                fnInc++;
        }

        if((abs(improvement) < INNER_THRESH && iters > MIN_INNER_ITER) || 
			iters > MAX_INNER_ITER || localeta < MIN_ETA )
                converge = true;

        if(fnInc>=1) {
                fprintf(stderr,"TOO MANY INCREASES !! I QUIT NOW :-(\n");
                return;
        }
    } 

}



void predict()
{
	double loss=0;
	vector<double> taskLoss(num_tasks);
	// squared error
	for( int t=0; t<num_tasks; t++ )
	{
	  for( int e=testTaskStartIndices[t]; e<testTaskStartIndices[t+1]; e++ )
	  {
		int nI = nodeToIdx_x[nodeI[e]];	
		int nJ = nodeToIdx_y[nodeJ[e]];	
		double predIJ=0;
		for( int k=0; k<numFactors; k++ )
			predIJ += XU[nI][k] * YV[nJ][k];
		double mtlpred = betas[t]*predIJ + (1-betas[t])*XSY[t][nI][nJ];
		taskLoss[t] += pow(edgeWt[e] - mtlpred,2);
		cout << "Edge: " << e << " true: " << edgeWt[e] << " pred: " << mtlpred << endl;
	  }
	  loss += taskLoss[t];
	  double taskSize = testTaskStartIndices[t+1]-testTaskStartIndices[t];
	  fprintf(stderr,"Task-%d  Loss: %.10g Avg-Loss: %.10g \n",t,taskLoss[t],taskLoss[t]/taskSize);
	}

	cerr << "Error: " << loss << " Avg.Error: " << (loss/(double)numEdges) << endl;
}

void save_model(string modelFile)
{
        string ufile = modelFile + ".U.mod";
        string vfile = modelFile + ".V.mod";
        ofstream outfile;

	if(numFactors>0) {
        outfile.open (ufile.c_str());
        for( int i = 0 ; i < d_x; i++ ) {
                for( int j = 0 ; j < numFactors-1; j++ )
                        outfile << U[i][j] << ",";
                outfile << U[i][numFactors-1] << endl;
        }
        outfile.close();

        outfile.open (vfile.c_str());
        for( int i = 0 ; i < d_y; i++ ) {
                for( int j = 0 ; j < numFactors-1; j++ )
                        outfile << V[i][j] << ",";
                outfile << V[i][numFactors-1] << endl;
        }
        outfile.close();
	}

		for( int t=0; t<num_tasks; t++) {
			string sfile = modelFile + "_task"+ to_string(t+1) + ".S.mod";
	        outfile.open (sfile.c_str());
    	    for( int i = 0 ; i < d_x; i++ ) {
        	        for( int j = 0 ; j < d_y-1; j++ )
            	            outfile << S[t][i][j] << ",";
                	outfile << S[t][i][d_y-1] << endl;
			}
        	outfile.close();
        }

}

void read2dMatrix(const char* filename, vector< vector <double> > &mat, int rows, int cols)
{
    printf("Reading 2D matrix from file: %s size: %dx%d\n",filename,rows,cols);
    std::ifstream file(filename);
    std::string line;
    int r=0;
    while(std::getline(file, line) && r < rows) {
        std::istringstream iss(line);
        std::string val;
	int c=0;
	while(std::getline(iss, val, ',') && c < cols) {
	 	mat[r][c++] = atof(val.c_str());
	}
	r++;
    }
    file.close();
}

void create_matrix(vector< vector <double> > &mat, int rows, int cols)
{
  for( int i = 0 ; i < rows; i++ )
  {
  	vector <double> temp(cols);
    mat.push_back(temp);
  }
}

void create_matrix_3d(vector< vector< vector <double> > > &mat, int rows, int cols, int more)
{
  for( int i = 0 ; i < rows; i++ ) {
    vector< vector <double> > first;
  	for( int j = 0 ; j < cols; j++ ) {
		vector <double> second(more);
		first.push_back(second);
	}
    mat.push_back(first);
  }
}


void read_data()
{
	try
	{
    	const char *featsFileX, *featsFileY;
		config.lookupValue("featsFileX",featsFileX);
		config.lookupValue("featsFileY",featsFileY);

        CLibSVMFile* featsFH_x = new CLibSVMFile(featsFileX,'r',"x");
        CLibSVMFile* featsFH_y = new CLibSVMFile(featsFileY,'r',"y");
		trainFeatsX = new CSparseFeatures<double>();
		trainFeatsY = new CSparseFeatures<double>();
        SGVector<double> nodesX = trainFeatsX->load_with_labels(featsFH_x, false);
        SGVector<double> nodesY = trainFeatsY->load_with_labels(featsFH_y, false);
		d_x = trainFeatsX->get_num_features();
		d_y = trainFeatsY->get_num_features();
		printf("[INFO] Finished reading features.. d_x: %d  d_y: %d\n",d_x,d_y);
		for( int idx=0; idx<nodesX.size(); idx++ )
			nodeToIdx_x[(int)nodesX[idx]] = idx;
		n_x = nodesX.size();
		for( int idx=0; idx<nodesY.size(); idx++ )
			nodeToIdx_y[(int)nodesY[idx]] = idx;
		n_y = nodesY.size();
		printf("[INFO] Number of nodes: %d x %d\n",n_x,n_y);

        featsFH_x->close();
        featsFH_y->close();
	}
	catch(const SettingNotFoundException &nfex)
	{
	    std::cerr << "[ReadData] No param setting in config file." << endl;
	}
}

void init_random(vector< vector <double> > &mat, int nrows, int ncols)
{
	for( int i = 0 ; i < nrows; i++ ) {
		for( int j = 0 ; j < ncols; j++ ) {
			mat[i][j] = (double)(rand()-rand())/(double)RAND_MAX / 10.0 ;
		}
	}
}

bool fileExists(const char *fileName)
{
    ifstream infile(fileName);
    return infile.good();
}

void init_parameters(int k, double l, double div)
{
	numFactors = k;
	lambda = l;
	srand( time( NULL ) );
	cost_factor = numEdges/div;

	// read sigmas
	string sigmasStr, temp;
	config.lookupValue("sigmas",sigmasStr);
	stringstream ss(sigmasStr);
	int c=0;
	printf("[INFO] Sigmas: ");
	while(!ss.eof()) {
		ss >> temp;
		sigmas.push_back(atof(temp.c_str()));
		printf("%.10g ",sigmas[c++]);
	}

	// read betas
	string betasStr;
	c=0;
	config.lookupValue("betas",betasStr);
	stringstream ssbeta(betasStr);
	printf("\n[INFO] Betas: ");
	while(!ssbeta.eof()) {
		ssbeta >> temp;
		betas.push_back(atof(temp.c_str()));
		printf("%.10g ",betas[c++]);
	}


	// read task indices
	string taskStr;
	config.lookupValue("taskIndices",taskStr);
	stringstream sst(taskStr);
	while(!sst.eof()) {
		sst >> temp;
		taskStartIndices.push_back(atoi(temp.c_str()));
	}
	num_tasks = taskStartIndices.size();
	printf("\n[INFO] Number of tasks: %d\t Task-indices: %s\n",num_tasks,taskStr.c_str());
	taskStartIndices.push_back(numEdges); // adding extra index to make it easier for loops

	// read test task indices
	taskStr;
	config.lookupValue("testTaskIndices",taskStr);
	stringstream testss(taskStr);
	int testNumEdges=0;
	while(!testss.eof()) {
		testss >> temp;
		int testEdges = atoi(temp.c_str());
		testNumEdges += testEdges;
		testTaskStartIndices.push_back(testEdges);
	}
	printf("[TEST-INFO] Number of tasks: %d\t Task-indices: %s\n",num_tasks,taskStr.c_str());
	testTaskStartIndices.push_back(testNumEdges); // adding extra index to make it easier for loops

	create_matrix(U, d_x, numFactors);
	create_matrix(V, d_y, numFactors);
	create_matrix_3d(S, num_tasks, d_x, d_y);

	int nz_count=0;
	/*for( int i = 0 ; i < num_tasks; i++ ) {
		for( int j = 0 ; j < d_x; j++ ) {
			for(int k = 0; k < d_y; k++) {
				//S[i][j][k] = (double)(rand()-rand())/(double)RAND_MAX / 10.0 ; // TODO: initialize most of these with 0s ??
				//double prob = (double)(rand())/(double)RAND_MAX;
				//if(prob <=0.001) {
				//	S[i][j][k] = 1;
				//	nz_count++;
				//}
				//S[i][j][k] = 0.001;
			}
		}
	}*/
	printf("O Yeah! Finished initializing S .. num of NZs: %d\n",nz_count);

  try {
    eta = config.lookup("learn_rate");
	const char *Ufilename, *Vfilename, *Sfilepath;
	config.lookupValue("initUfile",Ufilename);
	if(numFactors>0) {
		if(fileExists(Ufilename)) {
			read2dMatrix(Ufilename, U, d_x, k);
			printf("U : initial values [0,0] and [%d,%d] are %.10g %.10g\n",(d_x-1),(k-1),U[0][0],U[d_x-1][k-1]);
		}
		else
			printf("ERROR: %s file does not exist!\n",Ufilename);
	}
	else
		init_random(U, d_x, k);

	config.lookupValue("initVfile",Vfilename);
	if(numFactors>0 && fileExists(Vfilename)) {
		read2dMatrix(Vfilename, V, d_y, k);
		printf("V : initial values [0,0] and [%d,%d] are %.10g %.10g\n",(d_y-1),(k-1),V[0][0],V[d_y-1][k-1]);
	}
	else 
		init_random(V, d_y, k);

	config.lookupValue("initSfilepath",Sfilepath);
	if(fileExists(Sfilepath)) {
		char Staskfile[200];
		for( int t = 0 ; t < num_tasks; t++ ) {
			sprintf(Staskfile,"%s_task%d.S.mod",Sfilepath,(t+1));
			read2dMatrix(Staskfile, S[t], d_x, d_y);
			printf("S : initial values [0,0] and [%d,%d] are %.10g %.10g\n",(d_x-1),(d_y-1),S[t][0][0],S[t][d_x-1][d_y-1]);
		}
	}
  }
  catch(const SettingNotFoundException &nfex) {
     cerr << "[InitParams] No param setting in configuration file." << endl;
  }

	create_matrix(gradU, d_x, k);
	create_matrix(gradV, d_y, k);
	create_matrix_3d(gradS, num_tasks, d_x, d_y);
	create_matrix(XU, n_x, k);
	create_matrix(YV, n_y, k);
	create_matrix_3d(XSY, num_tasks, n_x, n_y);
	create_matrix(XgradU, n_x, k);
	create_matrix(YgradV, n_y, k);
	create_matrix_3d(XgradSY, num_tasks, n_x, n_y);

	compute_intermediate();
}

void free_parameters()
{
	U.clear();
	gradU.clear();
	V.clear();
	gradV.clear();
	S.clear();
	gradS.clear();
	XgradU.clear();
	XU.clear();
	YgradV.clear();
	YV.clear();
	XSY.clear();
	XgradSY.clear();
}

int readEdges(const char* matrixFile)
{
	// clear matrices
	nodeI.clear();
	nodeJ.clear();
	edgeWt.clear();
	// read file
        ifstream matrix(matrixFile);
	string line;
	int i,j; double val;
	while(getline(matrix, line)) {
		stringstream   linestream(line);
		linestream >> i >> j >> val;
		nodeI.push_back(i);
		nodeJ.push_back(j);
		edgeWt.push_back(val);
	}
	matrix.close();
	return edgeWt.size();
}

void read_edge_costs(const char* file)
{
	double c;
	if(file) {
		ifstream costs(file);
		if(costs.is_open()) {
			for(int i=0;i<=numEdges; i++) {
				costs >> c;
				cost_vec.push_back(c);
			}
		}
		cerr << "Finished reading costs.. from: " << file << endl;
	}
	else {
		for(int i=0;i<=numEdges; i++) {
			cost_vec[i]=1.0;
		}
	}
}

int main(int argc, char **argv)
{
        init_shogun(&print_message, &print_warning,&print_error);
        sg_io->set_loglevel(MSG_DEBUG);

        if(argc < 6) {
                printf("Insufficient arguments!\nUsage: ./bsl_mtl <config-file> <trainMatrix> <lambda> <num_factors_K> <model_prefix> <costFile> <cost> <l1/l2> <testFile> <iter-count>\n");
                exit(1);
        }

    const char* configFile = argv[1];
    const char* trainFile = argv[2];
	const double param_lambda = atof(argv[3]);
	const int K = atoi(argv[4]);
    string modelFile = argv[5];
	const char* costFile = argv[6];
    const double denom = atoi(argv[7]);
	const char* norm = argv[8];
    const char* testFile;
	if(argc >= 11)
		testFile = argv[9];
    int iterCount = 0;
	if(argc >= 12) {
		iterCount = atoi(argv[10]); // which iter to continue from
	}
	const int MAX_ITER = 50;
	const double THRESH=0.000001;

	read_config_file(configFile);
	read_data();

	cerr << "Reading edges (training)..." << trainFile << endl;
	numEdges = readEdges(trainFile);
	printf("[INFO] Finished reading matrix.. Number of edges: %d\n",numEdges); 
	// read cost per edge
	read_edge_costs(costFile);

	printf("[INFO]Lambda: %.10g\t K: %d\n",param_lambda,K);	
    printf("[INFO] Max iters: %d\tThreshold: %.10g\n",MAX_ITER,THRESH);
	printf("[INFO] Constant used to penalize squared error (cost_factor): nE/%.10g\n",denom);
	cout << "[INFO] Normalization on S: " << norm << endl;

	norm_param = (strcmp(norm,"l1")==0) ? L1 : L2;
	init_parameters(K, param_lambda, denom);
	fprintf(stderr,"Finished init params ...\n");
	compute_fn_value();
	double loss=FLT_MAX, oldloss;
	double prev_improv = 1;
	int threshExceedCount = 0, funcIncCount = 0;
	for(int iter=0; iter<=MAX_ITER; iter++)
	{
		fprintf(stderr,"Minimizing w.r.t V ...\n");
		if(K > 0)
			least_squares_V();
		fprintf(stderr,"Minimizing w.r.t U ...\n");
		if(K > 0)
			least_squares_U();
		for(int task=0; task<num_tasks; task++)
		{
			fprintf(stderr,"Minimizing w.r.t S%d ...\n",task);
			least_squares_S(task);
		}
		oldloss = loss;
		loss = compute_fn_value();
		double improvement = (oldloss-loss)/oldloss;
		fprintf(stderr,"Global Iteration# %d   Improvement: %.10g\n",iter,improvement);
		if(abs(improvement) < THRESH) {
			threshExceedCount++;
			if(threshExceedCount >= 5) {
				fprintf(stderr,"CONVERGING due to improvement threshold exceed!\n");
				break;
			}
		}
		if(improvement < -0.00001)
			funcIncCount++;
		if(std::isnan(improvement) || funcIncCount > 5) {
			fprintf(stderr,"STOPPING due to NAN or Increasing function :-( :-(\n");
			if(testFile) {
				cerr << "Reading test data...\n";
				numEdges = readEdges(testFile);
				cout << "Test labels:\n";
				predict();
			}
			return 0;
		}
		if(iter%10==0) {
			char numstr[21];
			sprintf(numstr, "%d", iterCount+iter);
			string newmodel = modelFile + '-' + numstr;
			fprintf(stderr,"[INFO] Saving model number %d to file: %s\n",(iter+iterCount),newmodel.c_str());
			save_model(newmodel);
		}
	}

	if(testFile) {
		cerr << "Reading test data...\n";
		numEdges = readEdges(testFile);
		cout << "Test labels:\n";
		predict();
	}

	save_model(modelFile);

	free_parameters();

    //SG_UNREF(feats);
    exit_shogun();
}
