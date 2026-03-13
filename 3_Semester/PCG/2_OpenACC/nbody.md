# PCG projekt 2
- autor: xkucht11

## Měření výkonu (čas / 100 kroků simulace)

### Průběžné
|   N   | CPU [s]  | Step 0 [s] | Step 1 [s] |
|:-----:|----------|------------|------------|
|  4096 | 0.492139 |  0.106810  |  0.009178  |
|  8192 | 1.471328 |  0.209519  |  0.034122  |
| 12288 | 2.478942 |  0.312982  |  0.071569  |
| 16384 | 3.386801 |  0.467626  |  0.122820  |
| 20480 | 5.059240 |  0.583755  |  0.146025  |
| 24576 | 7.112179 |  0.699913  |  0.207206  |
| 28672 | 9.892856 |  1.054989  |  0.279945  |
| 32768 | 12.59829 |  1.213505  |  0.360462  |
| 36864 | 15.54297 |  1.368781  |  0.456613  |
| 40960 | 19.36099 |  1.527582  |  0.560833  |
| 45056 | 23.48723 |  2.128217  |  0.672099  |
| 49152 | 27.69359 |  2.323011  |  0.797404  |
| 53248 | 32.63063 |  2.517049  |  0.937928  |
| 57344 | 37.43660 |  3.266181  |  1.083140  |
| 61440 | 42.85863 |  3.499262  |  1.239271  |
| 65536 | 49.46104 |  3.730787  |  1.405660  |
| 69632 | 55.14939 |  4.586002  |  1.601079  |
| 73728 | 62.04446 |  4.883218  |  1.781506  |
| 77824 | 69.26138 |  5.155460  |  1.987973  |
| 81920 | 76.60071 |  5.426927  |  2.190663  |


### Závěrečné
|    N   |  CPU [s] | GPU [s] | Zrychlení | Propustnost [GiB/s] | Výkon [GFLOPS] |
|:------:|:--------:|:-------:|:---------:|:-------------------:|:--------------:|
|   1024 |   1.0928 | 0.02086 |   52.39   |                     |                |
|   2048 |   0.5958 | 0.02406 |   24.77   |                     |                |
|   4096 |   0.6652 | 0.03124 |   21.30   |                     |                |
|   8192 |   1.6599 | 0.05422 |   30.57   |                     |                |
|  16384 |   3.3655 | 0.13713 |   24.54   |                     |                |
|  32768 |  12.7233 | 0.43974 |   28.94   |                     |                |
|  65536 |  48.9732 | 1.61651 |   30.30   |                     |                |
| 131072 | 195.9965 | 6.26490 |   31.29   |                     |                |

## Otázky

### Krok 0: Základní implementace
**Vyskytla se nějaká anomále v naměřených časech? Pokud ano, vysvětlete:**
mezi N = 24576 a 28672 (taky 53248, 57344) je významně větší časový skok běhu programu. pravděpodobně kvůli nedostatečnému počtu SM procesorů. Je taky možné že OpenAcc používa více registrů než CUDA implementace, a proto je počet bloků na SM omezen.

### Krok 1: Sloučení kernelů
**Došlo ke zrychlení?**
Ano, ~3x

**Popište hlavní důvody:**
1. Menší overhead nad spouštěním kernelů.
2. Nemusí se vícekrát počítat stejné proměnné v různých kernelech.

### Krok 2: Výpočet těžiště
**Kolik kernelů je nutné použít k výpočtu?**
1 (centerOfMass)

**Kolik další paměti jste museli naalokovat?**
16B (comBuffer)

**Jaké je zrychelní vůči sekveční verzi? Zdůvodněte.** *(Provedu to smyčkou #pragma acc parallel loop seq)*

pro N = 4096:
0.000430 / 0.000250 = 1.72x


### Krok 4: Měření výkonu
**Jakých jste dosáhli výsledků?**
Zrychlení ~30x

**Lze v datech pozorovat nějaké anomálie?**
Pro malé N je výkon významně ovlivňován funkcí h5Helper.writeParticleData(recordNum);