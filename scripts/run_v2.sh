set -e
cd "C:/Users/Kerem/Desktop/makale-yeni-adnan"
echo "=== FAZ 2 v2 baslangic: $(date) ==="
echo ">>> F3 fuzyon ablasyonu (20 kosu)"; python experiments/f3_fusion.py
echo ">>> F4 benchmark (45 kosu, 5'i F3'ten paylasimli)"; python experiments/f4_benchmark.py
echo ">>> F2 graf (30 kosu, 10'u paylasimli)"; python experiments/f2_graph.py
echo ">>> F1b adil temel (CPU)"; python experiments/f1b_fair_baseline.py
echo "=== bitis: $(date) ==="
