#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define NB_PAS_MAX 81
#define TPS_MAX_PAR_TEST 1
#define MIN_CLCK 1
#define TYPEREEL float


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


__global__ void initTab(TYPEREEL* tab, TYPEREEL val, int nbElt)
{
	for (int idx = 0; idx < nbElt; idx++)
		tab[idx] = val+idx;
}

__global__ void addTestLoop(TYPEREEL* tabA, TYPEREEL* tabB, TYPEREEL* tabC, int nbElt, int nbTest)
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

	size_t free, total;
	CUdevice dev;
	CUcontext ctx;
	cuInit(0);
	cuDeviceGet(&dev, 0);
	cuCtxCreate(&ctx, 0, dev);
	CUresult cuRes = cuMemGetInfo(&free, &total);
	if (cuRes != 0)
		std::cout << "Error: " << cudaGetErrorString(cudaError_t(cuRes)) << "\n";
	erreur = cudaGetLastError();
	if (erreur != cudaSuccess)
		std::cout << "Error: " << cudaGetErrorString(erreur) << "\n";
	std::cout << "free memory : " << free << "\n";
	std::cout << "total memory : " << total << "\n";

	cudaGetDeviceCount(&deviceCount);

	cudaGetDevice(&deviceId);
	cudaSetDevice(deviceId);
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);



	int nbPas = int(std::log(std::min(int(free / sizeof(TYPEREEL) / 3 /4), int(pow(2.0, NB_PAS_MAX / 3.0)))) / std::log(2))*3;
	if (nbPas <= 0)
	{
		std::cout << "Something went wrong...";
		return -1;
	}
	std::ofstream rapport("tps_fct_mem.txt");
	double tpsPre = 0;
	long long int nbTest = 1;
	int nbEltMax = int(pow(2.0, nbPas / 3.0));
	TYPEREEL*tabA, *tabB, *tabC;
	cudaMallocManaged(&tabA, sizeof(TYPEREEL) * nbEltMax);
	erreur = cudaGetLastError();
	if (erreur != cudaSuccess)
		std::cout << "Error: " << cudaGetErrorString(erreur) << "\n";
	cudaMallocManaged(&tabB, sizeof(TYPEREEL) * nbEltMax);
	erreur = cudaGetLastError();
	if (erreur != cudaSuccess)
		std::cout << "Error: " << cudaGetErrorString(erreur) << "\n";
	cudaMallocManaged(&tabC, sizeof(TYPEREEL) * nbEltMax);
	erreur = cudaGetLastError();
	if (erreur != cudaSuccess)
		std::cout << "Error: " << cudaGetErrorString(erreur) << "\n";
	double tps = 0;
	while (tps< MIN_CLCK)
	{
		nbTest *= 2;
		int nbElt = int(pow(2.0, 7 / 3.0));
		initTab << <1, 1 >> > (tabA, 2.0, nbElt);
		erreur = cudaGetLastError();
		if (erreur != cudaSuccess)
			std::cout << "Error: " << cudaGetErrorString(erreur) << "\n";
		initTab << <1, 1 >> > (tabB, 3.0, nbElt);
		erreur = cudaGetLastError();
		if (erreur != cudaSuccess)
			std::cout << "Error: " << cudaGetErrorString(erreur) << "\n";
		initTab << <1, 1 >> > (tabC, 0, nbElt);
		erreur = cudaGetLastError();
		if (erreur != cudaSuccess)
			std::cout << "Error: " << cudaGetErrorString(erreur) << "\n";
		double finPre;
		double debut = getCpuTime();
		addTestLoop << <1, 1 >> > (tabA, tabB, tabC, nbElt, nbTest);
		cudaDeviceSynchronize();
		erreur = cudaGetLastError();
		if (erreur != cudaSuccess)
			std::cout << "Error: " << cudaGetErrorString(erreur) << "\n";
		finPre = getCpuTime();
		tps = finPre - debut;

	}
	std::cout << "loop of" << nbTest <<std::endl;
	for (int idx = 7; idx < nbPas;idx++)
	{
		
		int nbElt = int(pow(2.0, idx / 3.0));
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
		double finPre;
		double debut = getCpuTime();
		addTestLoop <<<1, 1 >>> (tabA, tabB, tabC, nbElt, nbTest);
		erreur = cudaGetLastError();
		if (erreur != cudaSuccess)
			std::cout << "Error: " << cudaGetErrorString(erreur) << "\n";
		cudaDeviceSynchronize();
		erreur = cudaGetLastError();
		if (erreur != cudaSuccess)
			std::cout << "Error: " << cudaGetErrorString(erreur) << "\n";
		finPre = getCpuTime();
		double tps = finPre - debut;
		tpsParTest = tps;
		std::cout << "<-- " << nbElt * 3 * sizeof(TYPEREEL) << ", "  << nbElt << ", " << idx <<" -->\nDurée sans thread (" << tpsParTest << " ticks) ";
		tpsPre = tps;
		tpsParTest = tpsParTest  / nbTest;
		std::cout << tpsParTest << "s (" << tpsParTest / nbElt << "s par élément) nbTest=" << nbTest<<"\n";
		rapport << nbElt*3*sizeof(TYPEREEL) << "\t" << nbTest << "\t";
		rapport << tpsParTest / nbElt << "\t" << tps;
		rapport << "\n";
		rapport.flush();

	}
	cudaFree(tabA);
	cudaFree(tabB);
	cudaFree(tabC);
	rapport.close();
	std::cout << "\nEND\n";
    return 0;
}

