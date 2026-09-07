set -e
cd "C:/Users/Kerem/Desktop/makale-yeni-adnan"
echo "=== FAZ 4/3/5 baslangic: $(date) ==="
echo ">>> F5 CV + fuzyon ablasyonu (50 kosu)"; python experiments/f5_cv.py
echo ">>> F7 trip duzeyi istatistik";          python experiments/f7_cluster_stats.py
echo ">>> F6 verimlilik";                      python experiments/f6_efficiency.py
echo ">>> F9 figurler";                        python experiments/f9_figures_en.py
echo ">>> F9 mimari figuru";                   python experiments/f9_architecture.py
echo "=== bitis: $(date) ==="
