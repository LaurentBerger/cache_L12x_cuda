#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define NB_ELT 2
#define NB_TEST 1000000ll
#define NB_PAS_MAX 81
#define TPS_MAX_PAR_TEST 10
#define MIN_CLCK 1


#ifdef _WIN32
#include <windows.h>
#include <processthreadsapi.h>
FILETIME a, b, c, d;
inline double getCpuTime()
{
	if (GetProcessTimes(GetCurrentProcess(), &a, &b, &c, &d) != 0)
    {
		return
			(double)(d.dwLowDateTime |
				((unsigned long long)d.dwHighDateTime << 32)) * 0.0000001;
	}
	return 0;
}
#else
inline double getCpuTime()
{
	return std::clock() / double(CLOCKS_PER_SEC);
}
#endif


__global__ void initTab(double* tab, double val, int nbElt)
{
	for (int idx = 0; idx < nbElt; idx++)
		tab[idx] = val+idx;
}

__global__ void addTestLoop(double* tabA, double* tabB, double* tabC, int nbElt, int nbTest)
{
	for (long long int idxTest = 0; idxTest < nbTest; idxTest++)
		for (int i = 0; i < nbElt; i++)
			tabC[i] = tabA[i] + tabB[i];

}
 

int main() {
	int deviceId;
	int numberOfSMs;
	int deviceCount = 0;
	cudaError_t erreur;
	cudaGetDeviceCount(&deviceCount);

	cudaGetDevice(&deviceId);
	cudaSetDevice(deviceId);
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
	size_t free, total;
	CUresult cuRes=cuMemGetInfo(&free, &total);
	erreur = cudaGetLastError();
	if (erreur != cudaSuccess)
		std::cout << "Error: " << cudaGetErrorString(erreur) << "\n";
	std::cout << "free memory : " << free << "\n";
	std::cout << "total memory : " << total << "\n";


	int nbPas = int(std::log(std::min(int(free / 48), int(pow(2.0, NB_PAS_MAX / 3.0)))) / std::log(2))*3;
	std::ofstream rapport("tps_fct_mem.txt");
	double tpsPre = 0;
	long long int nbTest = NB_TEST;
	int nbEltMax = int(pow(2.0, nbPas / 3.0));
	double *tabA, *tabB, *tabC;
	cudaMallocManaged(&tabA, sizeof(double) * nbEltMax);
	erreur = cudaGetLastError();
	if (erreur != cudaSuccess)
		std::cout << "Error: " << cudaGetErrorString(erreur) << "\n";
	cudaMallocManaged(&tabB, sizeof(double) * nbEltMax);
	erreur = cudaGetLastError();
	if (erreur != cudaSuccess)
		std::cout << "Error: " << cudaGetErrorString(erreur) << "\n";
	cudaMallocManaged(&tabC, sizeof(double) * nbEltMax);
	erreur = cudaGetLastError();
	if (erreur != cudaSuccess)
		std::cout << "Error: " << cudaGetErrorString(erreur) << "\n";
	for (int idx = 7; idx < nbPas;idx++)
	{
		
		int nbElt = NB_ELT * int(pow(2.0, idx / 3.0));
		initTab <<<1, 1 >>> (tabA, 2.0, nbElt);
		erreur = cudaGetLastError();
		if (erreur != cudaSuccess)
			std::cout << "Error: " << cudaGetErrorString(erreur) << "\n";
		initTab <<<1, 1 >>> (tabB, 3.0, nbElt);
		erreur = cudaGetLastError();
		if (erreur != cudaSuccess)
			std::cout << "Error: " << cudaGetErrorString(erreur) << "\n";
		initTab <<<1, 1 >>> (tabC, 0, nbElt);
		erreur = cudaGetLastError();
		if (erreur != cudaSuccess)
			std::cout << "Error: " << cudaGetErrorString(erreur) << "\n";
		double tpsParTest = 0;
		if (tpsPre > TPS_MAX_PAR_TEST)
			nbTest /= 2;
		if (nbTest == 0)
			nbTest = 1;
		rapport << nbElt << "\t" << nbTest << "\t";
		double finPre;
		double debut = getCpuTime();
		double tpsMin = DBL_MAX;
		addTestLoop <<<1, 1 >>> (tabA, tabB, tabC, nbElt, nbTest);
		cudaDeviceSynchronize();
		erreur = cudaGetLastError();
		if (erreur != cudaSuccess)
			std::cout << "Error: " << cudaGetErrorString(erreur) << "\n";
		finPre = getCpuTime();
		double tps = finPre - debut;
		tpsParTest = tps;
		std::cout << "<-- " << nbElt << " -->\nDurée sans thread (" << tpsParTest << " ticks) ";
		tpsPre = tps;
		tpsParTest = tpsParTest  / nbTest;
		std::cout << tpsParTest << "s (" << tpsParTest / nbElt << "s par élément) nbTest=" << nbTest<<"\n";
		rapport << tpsParTest / nbElt << "\t" << tps << "\t" << tpsMin / nbElt ;
		rapport << "\n";
		rapport.flush();

	}
	delete tabA;
	delete tabB;
	delete tabC;
	rapport.close();
    return 0;
}

