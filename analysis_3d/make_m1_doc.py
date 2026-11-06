"""
M1 (Veri Kumesi Sayilarinda Tutarsizlik) icin hocaya gonderilecek
bagimsiz degerlendirme dokumani uretir. Manuscript docx'e DOKUNMAZ.

Tum sayilar gercek veri dosyalariyla dogrulanmistir:
  - analysis_3d/results_3d.csv          (293 satir = 293 trajektori)
  - data/training_data_particle_avg.csv (240 satir = 240 parcacik)
  - source_videos/ klasor yapisi        (161 sayisinin fiziksel teyidi)
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")

from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

ACCENT = RGBColor(0x1F, 0x4E, 0x79)   # koyu mavi (reviewer dokumaniyla uyumlu)
OK = RGBColor(0x2E, 0x7D, 0x32)       # yesil
BAD = RGBColor(0xC6, 0x28, 0x28)      # kirmizi

doc = Document()

# --- Stil: govde fontu ---
normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(11)


def heading(text, size=15, color=ACCENT, space_before=12):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(size)
    r.font.color.rgb = color
    return p


def body(text, italic=False, bold=False, space_after=6):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    r = p.add_run(text)
    r.italic = italic
    r.bold = bold
    return p


# ===========================================================================
# Baslik
# ===========================================================================
title = doc.add_paragraph()
tr = title.add_run("M1 — Veri Kümesi Sayılarında Tutarsızlık")
tr.bold = True
tr.font.size = Pt(18)
tr.font.color.rgb = ACCENT
sub = doc.add_paragraph()
sr = sub.add_run("Güncel verilerle yeniden değerlendirme ve düzeltme")
sr.italic = True
sr.font.size = Pt(12)
sr.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

body(
    "Reviewer'ın işaret ettiği 6 farklı \"N\" değeri tutarsız değildir; bunlar tek bir "
    "veri akışının ardışık aşamalarıdır. Aşağıdaki tüm sayılar projenin gerçek veri "
    "dosyalarından (results_3d.csv = 293 satır; training_data_particle_avg.csv = 240 satır) "
    "ve ham video klasör yapısından doğrulanmıştır. İki değer (105 ve 224) hatalıydı; "
    "kök nedenleri tespit edilip düzeltilmiştir.",
    space_after=10,
)

# ===========================================================================
# Tablo: Guncel N degerleri
# ===========================================================================
heading("1. Güncel ve doğrulanmış N değerleri", size=14)

cols = ["Değer", "Bulunduğu yer", "Anlamı", "Durum"]
rows = [
    ("240", "Abstract, Bölüm 2.1, ML", "Üretilen benzersiz parçacık (kategori + kod)", "Doğru"),
    ("720", "Bölüm 2.2", "240 × 3 tekrar = maksimum deney sayısı (\"up to 720\")", "Doğru"),
    ("348", "Bölüm 2.3", "Eşleştirilen MAK + ANG kamera çifti (ara aşama)", "Doğru"),
    ("293", "Abstract, Bölüm 2.3", "Başarılı 3D rekonstrüksiyon (%84.2 başarı)", "Doğru"),
    ("161", "Tablo 1, 2, 4; Bölüm 3.1", "Benzersiz parçacık (3D analizi başarılı)", "Doğru"),
    ("105 → 161", "Tablo 1, 2, 4; Bölüm 3.1", "Sayma hatası düzeltildi (aşağıda açıklandı)", "DÜZELTİLDİ"),
    ("224 → 293", "Şekil 8 (CD plot)", "Eski veri sürümünden kalmıştı", "DÜZELTİLDİ"),
]

table = doc.add_table(rows=1, cols=len(cols))
table.style = "Light Grid Accent 1"
hdr = table.rows[0].cells
for i, c in enumerate(cols):
    hdr[i].text = c
    for p in hdr[i].paragraphs:
        for run in p.runs:
            run.bold = True

for val, loc, mean, status in rows:
    cells = table.add_row().cells
    cells[0].text = val
    cells[1].text = loc
    cells[2].text = mean
    cells[3].text = status
    # Deger hucresini kalin yap
    for p in cells[0].paragraphs:
        for run in p.runs:
            run.bold = True
    # Durum rengini ayarla
    for p in cells[3].paragraphs:
        for run in p.runs:
            run.bold = True
            run.font.color.rgb = OK if status == "Doğru" else BAD

doc.add_paragraph()

# ===========================================================================
# Veri akisi
# ===========================================================================
heading("2. Aşamalar arası ilişki (veri akışı)", size=14)
body(
    "Aşağıdaki akış, sayıların neden farklı olduğunu açıklar. Reviewer'ın önerdiği "
    "PRISMA-benzeri akış şeması (Şekil 2) tam olarak bunu görselleştirecektir.",
    space_after=8,
)

flow_lines = [
    ("240 üretilen parçacık", "(4 polimer × 7 şekil)"),
    ("↓  × 3 tekrar", ""),
    ("≤ 720 settling deneyi", "→  ML modeli (240 parçacık → R² = 0.859)"),
    ("↓  çift-kamera (MAK + ANG) eşleştirme", ""),
    ("348 eşleştirilmiş kamera çifti", ""),
    ("↓  3D rekonstrüksiyon", "(%84.2 başarı / %15.8 başarısız: parçacık görüş alanından çıktı)"),
    ("293 başarılı 3D trajektori", "←  Tablo 1–2 ve Şekil 3–8'in temeli"),
    ("↓  benzersiz parçacığa indirgeme (kod + polimer)", ""),
    ("161 benzersiz parçacık", "(6 şekil; küre 3D analizine dahil değil)"),
]
for main, note in flow_lines:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(main)
    r.bold = main[0].isdigit()
    r.font.name = "Consolas"
    r.font.size = Pt(11)
    if main[0].isdigit():
        r.font.color.rgb = ACCENT
    if note:
        rn = p.add_run("   " + note)
        rn.italic = True
        rn.font.size = Pt(10)
        rn.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

doc.add_paragraph()
body(
    "240 ile 161 neden farklı?  240 = tüm settling-hızı veri seti (tek kamera yeterli, küre dahil); "
    "161 = bunlardan çift-kamera 3D rekonstrüksiyonu başarılı olan alt küme. 161 parçacığın tamamının "
    "240'ın içinde olduğu doğrulanmıştır (0 eksik). İkisi de doğrudur; farklı amaçlara hizmet ederler.",
    space_after=10,
)

# ===========================================================================
# Iki duzeltmenin kok nedeni
# ===========================================================================
heading("3. İki hatalı değerin kök nedenleri", size=14)

body("105 → 161 (Tablo 1, 2, 4 ve Bölüm 3.1):", bold=True, space_after=2)
body(
    "Tablo üretim kodu parçacıkları yalnızca \"kod\"a göre tekilleştiriyordu. Ancak aynı kod (örn. "
    "\"C-1\") farklı polimerlerde fiziksel olarak FARKLI parçacıklara karşılık geliyor — boyutları ve "
    "yoğunlukları farklı (C-1: RESIN 1210, PA6 1457, PLA 1376 kg/m³). \"Kod\"a göre tekilleştirme bu "
    "ayrı parçacıkları tek sayıya çöktürerek 161 yerine 105 üretiyordu. Doğru anahtar (kod + polimer) "
    "kullanıldığında sayı 161'dir. NOT: Bu, reviewer'ın sandığı gibi \"eski veri\" değil; güncel veride "
    "üretilen bir kodlama hatasıdır ve düzeltilmiştir.",
    space_after=8,
)

body("224 → 293 (Şekil 8, CD plot):", bold=True, space_after=2)
body(
    "Şekil 8'in başlığındaki n = 224, makale yazımı sırasında daha küçük bir veri sürümünden "
    "kalmıştır. Güncel veriyle (results_3d.csv, 293 satır) figür yeniden üretildiğinde 293 nokta verir. "
    "Yani 224 gerçekten eski veriden kaynaklanıyordu.",
    space_after=10,
)

# ===========================================================================
# 161'in fiziksel teyidi
# ===========================================================================
heading("4. 161 sayısının fiziksel teyidi", size=14)
body(
    "161 sayısı yalnızca tablo hesabına değil, ham video kayıtlarına da dayanır. Örneğin \"CUBE-7\" "
    "kodu hem ayrı bir ABS küpü (Ekim 2024'te çekilmiş) hem de ayrı bir PLA küpü (Kasım 2024'te "
    "çekilmiş) olarak ayrı klasörlerde, ayrı videolarla mevcuttur. Numaralandırma her polimerde "
    "sıfırdan başladığı için aynı kod farklı parçacıkları gösterir. Bu da 161'in (153 değil) doğru "
    "olduğunu kanıtlar.",
    space_after=10,
)

# ===========================================================================
# Oneri
# ===========================================================================
heading("5. Öneri", size=14)
body(
    "• Tablo 1, 2, 4 ve Bölüm 3.1'deki \"105\" → \"161\" olarak güncellenmeli.\n"
    "• Şekil 8 başlığındaki \"n = 224\" → \"n = 293\" olarak güncellenmeli.\n"
    "• Reviewer'ın önerdiği PRISMA-benzeri veri akış şeması (Bölüm 2'deki akış) Şekil 2 olarak eklenmeli.",
    space_after=4,
)

# --- Dipnot ---
note = doc.add_paragraph()
note.paragraph_format.space_before = Pt(14)
nr = note.add_run(
    "Not: Bu doküman yalnızca M1 (sayım tutarlılığı) konusunu kapsar. Veri doğrulaması sırasında "
    "M1 dışında ayrı bir konu (bazı küp parçacıklarında yoğunluk değeri tutarsızlığı) da tespit "
    "edilmiştir; bu, sayım sorunundan bağımsız olup ayrıca değerlendirilecektir."
)
nr.italic = True
nr.font.size = Pt(9)
nr.font.color.rgb = RGBColor(0x77, 0x77, 0x77)

# ===========================================================================
out = os.path.join(os.path.expanduser("~"), "Downloads", "M1_Veri_Kumesi_Sayilari_Degerlendirme.docx")
doc.save(out)
print("Kaydedildi:", out)
