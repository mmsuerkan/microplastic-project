import os
import shutil
import sys
sys.stdout.reconfigure(encoding='utf-8')

src = r"D:\ECE'YE VERILECEK"
dst = r"C:\Users\mmert\PycharmProjects\ObjectTrackingProject\temp_frames"

print(f'Kaynak: {src}')
print(f'Kaynak var mi: {os.path.exists(src)}')

if os.path.exists(src):
    print('Kopyalama basliyor...')
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print('Kopyalama tamamlandi!')
else:
    print('Kaynak bulunamadi!')
