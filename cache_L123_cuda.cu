#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include  <cfloat>

#define NB_ELT 2
#define NB_TEST 1000000ll
#define NB_PAS 81
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
	int accessUM;
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);

	cudaGetDevice(&deviceId);
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
	cudaDeviceGetAttribute(&accessUM, cudaDevAttrConcurrentManagedAccess, deviceId);
	
	int maxMem = NB_ELT *1024 * 256;
	double pasMem = (std::log(maxMem) - std::log(NB_ELT))/ NB_PAS;
	std::ofstream rapport("tps_fct_mem.txt");
	double tpsPre = 0;
	long long int nbTest = NB_TEST;
	int nbEltMax = NB_ELT * int(pow(2.0, NB_PAS / 3.0));
	double *tabA, *tabB, *tabC;
	cudaMallocManaged(&tabA, sizeof(double) * nbEltMax);
	cudaMallocManaged(&tabB, sizeof(double) * nbEltMax);
	cudaMallocManaged(&tabC, sizeof(double) * nbEltMax);
	for (int idx = 7; idx < NB_PAS;idx++)
	{
		
		int nbElt = NB_ELT * int(pow(2.0, idx / 3.0));
		initTab <<<1, 1 >>> (tabA, 2.0, nbElt);
		initTab <<<1, 1 >>> (tabB, 3.0, nbElt);
		initTab <<<1, 1 >>> (tabC, 0, nbElt);
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

