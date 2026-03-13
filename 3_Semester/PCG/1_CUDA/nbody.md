# PCG projekt 1
- autor: xkucht11

## Měření výkonu (čas / 100 kroků simulace)

### Průběžné
|   N   |  CPU [s]  | Step 0 [s] | Step 1 [s] | Step 2 [s] |
|:-----:|-----------|------------|------------|------------|
|  4096 | 0.327809  | 0.375041   | 0.249975   | 0.230794s  |
|  8192 | 0.936767  | 0.753593   | 0.499568   | 0.458272s  |
| 12288 | 1.880344  | 1.129795   | 0.749551   | 0.685717s  |
| 16384 | 3.372790  | 1.505540   | 0.998955   | 0.913188s  |
| 20480 | 4.893386  | 1.880371   | 1.248443   | 1.140618s  |
| 24576 | 7.055286  | 2.256917   | 1.498188   | 1.367939s  |
| 28672 | 9.517616  | 2.633479   | 1.747391   | 1.595389s  |
| 32768 | 12.295328 | 3.008508   | 1.996977   | 1.822922s  |
| 36864 | 15.386642 | 3.385974   | 2.246998   | 2.050266s  |
| 40960 | 19.076958 | 3.761568   | 2.497554   | 2.277636s  |
| 45056 | 23.350172 | 4.136106   | 2.748141   | 2.505019s  |
| 49152 | 27.441240 | 4.511933   | 2.998586   | 2.732449s  |
| 53248 | 32.652786 | 4.887970   | 3.245180   | 2.959770s  |
| 57344 | 37.600342 | 8.299985   | 5.638919   | 5.105693s  |
| 61440 | 43.587978 | 8.899113   | 6.043338   | 5.471385s  |
| 65536 | 48.961647 | 9.504220   | 6.451826   | 5.835068s  |
| 69632 | 56.018734 | 10.099150  | 6.853933   | 6.199717s  |
| 73728 | 61.949821 | 10.694995  | 7.256051   | 6.563943s  |
| 77824 | 69.667244 | 11.294595  | 7.661868   | 6.928553s  |
| 81920 | 75.934677 | 11.887570  | 8.069431   | 7.293632s  |


### Závěrečné
|    N   |  CPU [s] |  GPU [s]  | Zrychlení | Propustnost [GiB/s] | Výkon [GFLOPS] |
|:------:|:--------:|:---------:|:---------:|:-------------------:|:--------------:|
|   1024 |   1.0928 | 0.070084  |  15.59    |                     |                |
|   2048 |   0.5958 | 0.130274  |  4.57     |                     |                |
|   4096 |   0.6652 | 0.248450  |  2.68     |                     |                |
|   8192 |   1.6599 | 0.488267  |  3.40     |                     |                |
|  16384 |   3.3655 | 0.964283  |  3.49     |                     |                |
|  32768 |  12.7233 | 1.918964  |  6.63     |                     |                |
|  65536 |  48.9732 | 6.025675  |  8.13     |                     |                |
| 131072 | 195.9965 | 18.093096 |  10.83    |       0.024         |     15100      |

## Otázky

### Krok 0: Základní implementace
**Vyskytla se nějaká anomále v naměřených časech? Pokud ano, vysvětlete:**
1) časy na gpu jsou horší než na cpu pro N = 4096. 
1. rychlejší CPU: Myslím, že hlavním důvodem bude schopnost uložení většiny dat na CPU do L1 cache.
2. pomalejší GPU: Načítání dat do a z GPU, a overhead nad spouštěním kernelů.

2) obrovský výkonovy skok 
53248:  4.887970
57344:  8.299985

A100 -> 108 SM processors (64 threads)

53248 / 512 = 104
57344 / 512 = 112 -> 2 iterace na některých SM kvůli jejich nedostatečnému počtu.
dalo by se zlepšit mírným zvětšením vláken na blok, aby byl průchod vždy jen 1.

(samozřejmě to jen posunuje problém na trochu vetší N a pri příliš velikých datech to nepomůže, protože např 2x 512 vláken je pro gpu výpočetně téměř to stejné jako 1x 1024 vláken)
navíc nevýhodou může být nezarovnané přistupování do paměti při použití "nepěkného" počtu vláken na blok.
Ale násobky 32 vláken by měli být v pořádku 32vláken * 4 floaty = 32x4B = 128B = velikost cache line


### Krok 1: Sloučení kernelů
**Došlo ke zrychlení?**
Ano, o ~30%.

**Popište hlavní důvody:**
1. Menší overhead nad spouštěním kernelů.
2. Nemusí se vícekrát počítat stejné proměnné v různých kernelech.

### Krok 2: Sdílená paměť
**Došlo ke zrychlení?**
Ano o ~10%.

**Popište hlavní důvody:**
Menší latence přístupu pro vícekrát používaná data.
Každopádně si myslím že algoritmus má poměrně velkou aritmetickou intenzitu, a proto použití sdílené paměti urychlilo výpočet jenom minimálně.

### Krok 5: Měření výkonu
**Jakých jste dosáhli výsledků?**

**Lze v datech pozorovat nějaké anomálie?**
Na straně GPU to stejné jak předtím, kdy výkon rapidně klesl pro N = 65536 a 131072 (Málo SM procesorů).
Na straně CPU je N = 1024 pomalejší než 2048 a 4096.
