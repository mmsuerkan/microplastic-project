"""
Generate a Word document report for the 3D Settling Dynamics Paper.
Includes all figures, tables, and analysis results.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_ORIENT

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
FIG_DIR = os.path.join(BASE_DIR, 'figures')
TABLE_DIR = os.path.join(BASE_DIR, 'tables')


def set_cell_shading(cell, color):
    """Set cell background color."""
    from docx.oxml.ns import qn
    shading = cell._element.get_or_add_tcPr()
    shading_elm = shading.makeelement(qn('w:shd'), {
        qn('w:fill'): color,
        qn('w:val'): 'clear',
    })
    shading.append(shading_elm)


def style_table(table):
    """Apply clean styling to a table."""
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(table.rows):
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in paragraph.runs:
                    run.font.size = Pt(9)
                    run.font.name = 'Times New Roman'
            if i == 0:
                set_cell_shading(cell, '2E4057')
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.color.rgb = RGBColor(255, 255, 255)
                        run.font.bold = True


def add_heading_styled(doc, text, level=1):
    """Add a styled heading."""
    heading = doc.add_heading(text, level=level)
    for run in heading.runs:
        run.font.name = 'Times New Roman'
    return heading


def generate_report():
    doc = Document()

    # Page margins
    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)

    # Default font
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(11)

    # ========== TITLE PAGE ==========
    doc.add_paragraph()
    doc.add_paragraph()

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run('Three-Dimensional Settling Dynamics and Machine Learning-Based '
                        'Velocity Prediction of Microplastic Particles with Controlled Geometries')
    run.font.size = Pt(18)
    run.font.bold = True
    run.font.name = 'Times New Roman'
    run.font.color.rgb = RGBColor(0x2E, 0x40, 0x57)

    doc.add_paragraph()

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run('Technical Report — 3D Trajectory Analysis Results')
    run.font.size = Pt(13)
    run.font.italic = True
    run.font.name = 'Times New Roman'

    doc.add_paragraph()

    info = doc.add_paragraph()
    info.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = info.add_run('Target Journal: Water Research / Marine Pollution Bulletin\n'
                       'Date: March 2026')
    run.font.size = Pt(11)
    run.font.name = 'Times New Roman'

    doc.add_page_break()

    # ========== TABLE OF CONTENTS ==========
    add_heading_styled(doc, 'Table of Contents', level=1)
    toc_items = [
        '1. Introduction and Objectives',
        '2. Methodology',
        '   2.1 Particle Preparation',
        '   2.2 Experimental Setup',
        '   2.3 Tracking Algorithm',
        '   2.4 3D Trajectory Reconstruction',
        '   2.5 Trajectory Metrics',
        '   2.6 Machine Learning Model',
        '3. Results',
        '   3.1 Particle Properties',
        '   3.2 3D Settling Trajectories',
        '   3.3 Drift and Tortuosity Analysis',
        '   3.4 Reynolds Number and Drag Coefficient',
        '   3.5 Machine Learning Prediction',
        '   3.6 Comparison with Lofty et al. (2026)',
        '4. Key Findings',
        '5. Conclusions',
    ]
    for item in toc_items:
        p = doc.add_paragraph(item)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.space_before = Pt(2)
        for run in p.runs:
            run.font.size = Pt(10)

    doc.add_page_break()

    # ========== 1. INTRODUCTION ==========
    add_heading_styled(doc, '1. Introduction and Objectives', level=1)

    doc.add_paragraph(
        'The presence of microplastic (MP) pollution in aquatic environments has profound implications '
        'for biodiversity, food security, and freshwater resources. Understanding how MP particles settle '
        'through the water column is essential for modeling their transport, predicting deposition patterns, '
        'and assessing environmental risks. The settling velocity of a particle is governed by its shape, '
        'size, density, and the balance between buoyant forces and drag.'
    )

    doc.add_paragraph(
        'While previous research has largely examined one-dimensional vertical settling velocities, '
        'very little is known about the three-dimensional settling dynamics of microplastics — including '
        'their lateral drifting, settling path tortuosity, and horizontal velocities. Recent work by '
        'Lofty et al. (2026) in Environmental Science & Technology investigated the 3D settling behavior '
        'of 127 environmental MP particles using a dual-camera system, revealing that elongated particles '
        'exhibit pronounced lateral movements reaching up to 65 times their equivalent diameter.'
    )

    doc.add_paragraph(
        'However, key gaps remain in the literature:'
    )

    gaps = [
        'Most 3D studies use environmental particles with unknown density — a critical parameter '
        'for settling behavior that Lofty et al. explicitly identified as a limitation.',
        'Controlled experiments with known geometries are needed to isolate the effect of particle '
        'shape on settling dynamics.',
        'No study has combined 3D trajectory analysis with machine learning-based velocity prediction.',
        'Statistical rigor is limited: most particles are tested only once, preventing quantification '
        'of settling velocity variance.',
    ]
    for gap in gaps:
        p = doc.add_paragraph(gap, style='List Bullet')
        for run in p.runs:
            run.font.size = Pt(11)

    doc.add_paragraph(
        'This study addresses these gaps by investigating the three-dimensional settling dynamics of '
        '240 microplastic particles with controlled geometries and known densities. Using a dual-camera '
        'system positioned at 90° angles, we reconstruct 3D trajectories for 293 paired experiments '
        'across 7 distinct shape types. We quantify settling velocities, horizontal drift, trajectory '
        'tortuosity, and oscillation amplitude, and develop a machine learning model to predict settling '
        'velocities from particle geometry alone.'
    )

    doc.add_page_break()

    # ========== 2. METHODOLOGY ==========
    add_heading_styled(doc, '2. Methodology', level=1)

    # 2.1
    add_heading_styled(doc, '2.1 Particle Preparation', level=2)
    doc.add_paragraph(
        'A total of 240 microplastic particles were fabricated from four polymer types: PA6 (polyamide, '
        'ρ ≈ 1146 kg/m³), PMMA (polymethyl methacrylate, ρ ≈ 1175 kg/m³), Resin (ρ ≈ 1007–1455 kg/m³), '
        'and WSP (water-soluble polymer). Particles were machined into seven distinct geometric shapes: '
        'Cylinder, Half Cylinder, Cube, Wedge Shape Prism, Box Shape Prism, Sphere, and Elliptic Cylinder. '
        'Three orthogonal dimensions (a, b, c) were measured using digital calipers for each particle.'
    )

    doc.add_paragraph(
        'The equivalent diameter was calculated as the geometric mean of the three axes:'
    )
    eq = doc.add_paragraph()
    eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = eq.add_run('Deq = (a × b × c)^(1/3)')
    run.font.italic = True
    run.font.size = Pt(11)

    # 2.2
    add_heading_styled(doc, '2.2 Experimental Setup', level=2)
    doc.add_paragraph(
        'Settling experiments were conducted in a cylindrical water column with an inner diameter of '
        '30 cm filled with deionized water at room temperature. Two cameras (50 fps) were positioned '
        'at 90° angles to each other, viewing the X–Z and Y–Z planes respectively. The cameras were '
        'designated as MAK (macro view, X–Z plane) and ANG (angled view, Y–Z plane). A calibration '
        'height of 28.5 cm was used for pixel-to-centimeter conversion.'
    )
    doc.add_paragraph(
        'Each particle was released beneath the water surface using fine forceps and allowed to reach '
        'an equilibrium settling state before entering the camera detection window. Each particle was '
        'tested three times (FIRST, SECOND, THIRD repeat) to ensure statistical rigor. The cylindrical '
        'column geometry provides symmetric wall effects, eliminating corner-induced asymmetries present '
        'in rectangular tanks.'
    )

    # 2.3
    add_heading_styled(doc, '2.3 Tracking Algorithm', level=2)
    doc.add_paragraph(
        'An automated particle tracking algorithm was developed using OpenCV. The algorithm employs '
        'background subtraction with CLAHE (Contrast Limited Adaptive Histogram Equalization) enhancement '
        'to detect the particle in each frame. The centroid of the detected contour is tracked frame-by-frame, '
        'producing pixel coordinates (X, Y) and frame-to-frame displacements (dX, dY, Magnitude) for each '
        'camera view independently.'
    )
    doc.add_paragraph(
        'The tracking system achieves an End Point Error (EPE) of 0.47 pixels on average, well below '
        'the target threshold of 3–4 pixels specified in the project requirements.'
    )

    # 2.4
    add_heading_styled(doc, '2.4 3D Trajectory Reconstruction', level=2)
    doc.add_paragraph(
        'Three-dimensional trajectories were reconstructed by combining the synchronized MAK and ANG '
        'camera recordings for each experiment. The MAK camera provides the particle\'s (x, z) coordinates '
        'in real-world space, while the ANG camera provides (y, z) coordinates. The vertical coordinate z '
        'is averaged from both cameras, following the method of Lofty et al. (2026).'
    )
    doc.add_paragraph(
        'Since the two cameras may record different numbers of frames due to the particle entering and '
        'exiting the detection window at different times, trajectories are aligned by normalizing the '
        'vertical position to [0, 1] and interpolating both cameras\' data onto a common grid. This '
        'produces a unified 3D trajectory (x, y, z) with estimated time and velocity components (vx, vy, vz).'
    )
    doc.add_paragraph(
        'Of 348 paired MAK+ANG experiments identified, 293 were successfully reconstructed in 3D. '
        'The remaining 55 pairs failed due to missing tracking data, particle property mismatches, '
        'or numerical convergence issues.'
    )

    # 2.5
    add_heading_styled(doc, '2.5 Trajectory Metrics', level=2)
    doc.add_paragraph(
        'The following trajectory metrics were computed for each particle, adapted from the framework '
        'of Lofty et al. (2026):'
    )

    metrics_desc = [
        ('Settling velocity (w)', 'Ensemble-averaged vertical velocity component: w = ⟨vz⟩'),
        ('Horizontal drift velocity (V)', 'Ensemble-averaged horizontal speed: V = ⟨√(vx² + vy²)⟩'),
        ('Drift (δ)', 'Euclidean distance between the particle\'s starting position and the endpoint '
         'of its average trajectory in the horizontal plane'),
        ('Normalized drift (δ*)', 'Drift divided by equivalent diameter: δ* = δ / Deq'),
        ('Amplitude (σ)', 'Standard deviation of lateral deviations from the average trajectory line'),
        ('Drift gradient (ε)', 'Ratio of lateral drift to total vertical distance: ε = δ / Δz'),
        ('Tortuosity (φ)', 'Percentage deviation of actual path length from straight-line distance: '
         'φ = ((L − D) / D) × 100'),
        ('Particle Reynolds number (Rep)', 'Rep = w × Deq / ν, where ν is kinematic viscosity of water'),
    ]

    for name, desc in metrics_desc:
        p = doc.add_paragraph()
        run_bold = p.add_run(f'{name}: ')
        run_bold.bold = True
        run_bold.font.size = Pt(11)
        run_normal = p.add_run(desc)
        run_normal.font.size = Pt(11)

    # 2.6
    add_heading_styled(doc, '2.6 Machine Learning Model', level=2)
    doc.add_paragraph(
        'A balanced ensemble model combining Random Forest (70% weight) and Neural Network (30% weight) '
        'was trained to predict settling velocity from particle geometry. The model uses 9 input features: '
        'three dimensions (a, b, c), density, shape encoding, volume, surface area, aspect ratio, and '
        'volume-to-surface-area ratio. Performance was evaluated using 5-fold cross-validation.'
    )
    doc.add_paragraph(
        'Additionally, we investigated whether adding 3D trajectory features (normalized drift, tortuosity, '
        'drift gradient) improves prediction performance, to determine whether settling dynamics contain '
        'information beyond what particle geometry already provides.'
    )

    doc.add_page_break()

    # ========== 3. RESULTS ==========
    add_heading_styled(doc, '3. Results', level=1)

    # 3.1 Particle Properties
    add_heading_styled(doc, '3.1 Particle Properties', level=2)

    # Load table 1
    t1 = pd.read_csv(os.path.join(TABLE_DIR, 'table1_particle_properties.csv'))

    doc.add_paragraph(
        'Table 1 summarizes the geometric properties of the 105 unique particles successfully analyzed '
        'in 3D across 6 shape categories. Equivalent diameters range from 1.84 to 9.25 mm, and particle '
        'densities range from 1,007 to 1,455 kg/m³.'
    )

    # Insert Table 1
    p_caption = doc.add_paragraph()
    p_caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p_caption.add_run('Table 1. Particle Properties Summary by Shape')
    run.bold = True
    run.font.size = Pt(10)

    table = doc.add_table(rows=len(t1) + 1, cols=len(t1.columns))
    # Headers
    for j, col in enumerate(t1.columns):
        table.cell(0, j).text = col
    # Data
    for i, row in t1.iterrows():
        for j, val in enumerate(row):
            table.cell(i + 1, j).text = str(val)
    style_table(table)

    doc.add_paragraph()

    # 3.2 3D Trajectories
    add_heading_styled(doc, '3.2 3D Settling Trajectories', level=2)

    doc.add_paragraph(
        'Figure 2 shows representative 3D settling trajectories for each of the six shape categories. '
        'Clear differences in settling behavior are observed: cylindrical and cubic particles follow '
        'relatively straight vertical paths, while elliptic cylinders and wedge-shaped particles '
        'exhibit pronounced lateral movements and more complex settling patterns.'
    )

    # Insert Figure 2
    fig2_path = os.path.join(FIG_DIR, 'figure2_3d_trajectories.png')
    if os.path.exists(fig2_path):
        doc.add_picture(fig2_path, width=Inches(5.5))
        last_paragraph = doc.paragraphs[-1]
        last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = cap.add_run(
            'Figure 2. Representative 3D settling trajectories for each particle shape category. '
            'Each trajectory shows the particle\'s path through the 28.5 cm detection window.'
        )
        run.font.size = Pt(9)
        run.font.italic = True

    doc.add_paragraph()

    # 3.3 Drift and Tortuosity
    add_heading_styled(doc, '3.3 Drift and Tortuosity Analysis', level=2)

    doc.add_paragraph(
        'Table 2 presents the 3D settling metrics grouped by particle shape. Significant variations '
        'were observed across shape categories:'
    )

    # Load results for inline stats
    df = pd.read_csv(os.path.join(BASE_DIR, 'results_3d.csv'))

    findings = [
        f'Settling velocities ranged from {df.settling_velocity_cm_s.min():.1f} to '
        f'{df.settling_velocity_cm_s.max():.1f} cm/s (mean: {df.settling_velocity_cm_s.mean():.1f} '
        f'± {df.settling_velocity_cm_s.std():.1f} cm/s).',

        f'Tortuosity values ranged from {df.tortuosity_pct.min():.1f}% to '
        f'{df.tortuosity_pct.max():.1f}%, with elliptic cylinders showing the highest '
        f'mean tortuosity (22.9 ± 10.8%) — nearly three times that of cylindrical particles (7.7 ± 4.5%).',

        f'Drift gradients ranged from {df.drift_gradient.min():.3f} to {df.drift_gradient.max():.3f}, '
        f'indicating that some particles travel laterally up to 36% of their vertical descent distance.',

        f'Horizontal drift velocities averaged {df.horizontal_drift_velocity_cm_s.mean():.1f} '
        f'± {df.horizontal_drift_velocity_cm_s.std():.1f} cm/s, representing '
        f'{(df.horizontal_drift_velocity_cm_s.mean() / df.settling_velocity_cm_s.mean() * 100):.0f}% '
        f'of the mean settling velocity.',
    ]
    for f_text in findings:
        doc.add_paragraph(f_text, style='List Bullet')

    doc.add_paragraph()

    # Insert Table 2
    t2 = pd.read_csv(os.path.join(TABLE_DIR, 'table2_settling_metrics.csv'))

    p_caption = doc.add_paragraph()
    p_caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p_caption.add_run('Table 2. 3D Settling Metrics by Particle Shape')
    run.bold = True
    run.font.size = Pt(10)

    table2 = doc.add_table(rows=len(t2) + 1, cols=len(t2.columns))
    for j, col in enumerate(t2.columns):
        table2.cell(0, j).text = col
    for i, row in t2.iterrows():
        for j, val in enumerate(row):
            table2.cell(i + 1, j).text = str(val)
    style_table(table2)

    doc.add_paragraph()

    # Insert Figure 3
    fig3_path = os.path.join(FIG_DIR, 'figure3_drift_tortuosity.png')
    if os.path.exists(fig3_path):
        doc.add_picture(fig3_path, width=Inches(6.0))
        last_paragraph = doc.paragraphs[-1]
        last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = cap.add_run(
            'Figure 3. Distribution of (a) normalized drift δ*, (b) tortuosity φ (%), and '
            '(c) drift gradient ε across particle shapes. Box plots show median, interquartile range, '
            'and outliers.'
        )
        run.font.size = Pt(9)
        run.font.italic = True

    doc.add_page_break()

    # 3.4 Re-CD
    add_heading_styled(doc, '3.4 Reynolds Number and Drag Coefficient', level=2)

    doc.add_paragraph(
        f'Particle Reynolds numbers ranged from {df.reynolds_number.min():.0f} to '
        f'{df.reynolds_number.max():.0f}, spanning the intermediate to turbulent settling regimes. '
        f'Figure 4 presents the drag coefficient (CD) as a function of Reynolds number (Rep) for all '
        f'293 experiments, colored by particle shape. The standard sphere drag curve is overlaid for '
        f'reference.'
    )

    doc.add_paragraph(
        'Non-spherical particles generally exhibit higher drag coefficients compared to the sphere curve, '
        'with the magnitude of deviation depending on particle shape. Elliptic cylinders and wedge-shaped '
        'particles show the greatest departure from spherical behavior, consistent with their higher '
        'tortuosity and lateral drift.'
    )

    # Insert Figure 4
    fig4_path = os.path.join(FIG_DIR, 'figure4_re_cd.png')
    if os.path.exists(fig4_path):
        doc.add_picture(fig4_path, width=Inches(5.0))
        last_paragraph = doc.paragraphs[-1]
        last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = cap.add_run(
            'Figure 4. Drag coefficient (CD) versus particle Reynolds number (Rep) for 293 experiments, '
            'colored by particle shape. The solid line represents the standard sphere drag curve.'
        )
        run.font.size = Pt(9)
        run.font.italic = True

    doc.add_paragraph()

    # 3.5 ML
    add_heading_styled(doc, '3.5 Machine Learning Prediction', level=2)

    doc.add_paragraph(
        'The ensemble machine learning model (Random Forest 70% + Neural Network 30%) achieved a '
        'cross-validated R² = 0.859 ± 0.035 using the full dataset of 240 particles with 9 geometric '
        'features. The train–test gap of 0.061 indicates low overfitting risk.'
    )

    doc.add_paragraph(
        'The most important features were aspect ratio (0.263), dimension a (0.162), dimension b (0.115), '
        'and volume-to-surface-area ratio (0.099), indicating that particle shape descriptors are more '
        'informative than raw dimensions for velocity prediction.'
    )

    # ML comparison table
    ml = pd.read_csv(os.path.join(BASE_DIR, 'ml_comparison.csv'))

    p_caption = doc.add_paragraph()
    p_caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p_caption.add_run('Table 3. Machine Learning Model Comparison: Baseline vs 3D-Enhanced')
    run.bold = True
    run.font.size = Pt(10)

    ml_table = doc.add_table(rows=len(ml) + 1, cols=len(ml.columns))
    for j, col in enumerate(ml.columns):
        ml_table.cell(0, j).text = col
    for i, row in ml.iterrows():
        for j, val in enumerate(row):
            ml_table.cell(i + 1, j).text = str(round(val, 4)) if isinstance(val, float) else str(val)
    style_table(ml_table)

    doc.add_paragraph()

    doc.add_paragraph(
        'Adding 3D trajectory features (normalized drift, tortuosity, drift gradient) to the model '
        'did not improve cross-validated performance (R² = 0.726 vs 0.738 baseline on the 161-particle '
        'subset). While train R² increased (0.877 vs 0.861), the features introduced additional noise '
        'and overfitting. This suggests that settling dynamics are already captured by the geometric '
        'features — the 3D trajectory metrics describe the consequences of particle geometry, not '
        'independent information.'
    )

    # Insert Figure 5
    fig5_path = os.path.join(FIG_DIR, 'figure5_ml_prediction.png')
    if os.path.exists(fig5_path):
        doc.add_picture(fig5_path, width=Inches(6.0))
        last_paragraph = doc.paragraphs[-1]
        last_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = cap.add_run(
            'Figure 5. (Left) Predicted vs measured settling velocity with 1:1 line and ±20% bands. '
            '(Right) Feature importance from the Random Forest component of the ensemble model.'
        )
        run.font.size = Pt(9)
        run.font.italic = True

    doc.add_page_break()

    # 3.6 Comparison with Lofty
    add_heading_styled(doc, '3.6 Comparison with Lofty et al. (2026)', level=2)

    doc.add_paragraph(
        'Table 4 presents a systematic comparison between this study and the recent work of '
        'Lofty et al. (2026), which investigated the 3D settling dynamics of 127 environmental '
        'microplastic particles.'
    )

    # Insert Table 3 (comparison)
    t3 = pd.read_csv(os.path.join(TABLE_DIR, 'table3_comparison_lofty.csv'))

    p_caption = doc.add_paragraph()
    p_caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p_caption.add_run('Table 4. Comparison with Lofty et al. (2026)')
    run.bold = True
    run.font.size = Pt(10)

    comp_table = doc.add_table(rows=len(t3) + 1, cols=len(t3.columns))
    for j, col in enumerate(t3.columns):
        comp_table.cell(0, j).text = col
    for i, row in t3.iterrows():
        for j, val in enumerate(row):
            comp_table.cell(i + 1, j).text = str(val)
    style_table(comp_table)

    doc.add_paragraph()

    advantages = [
        'Known particle density: Lofty et al. explicitly acknowledge the absence of density measurements '
        'as a limitation. Our study includes measured densities for all particles, enabling drag coefficient '
        'calculations and more physically grounded analysis.',

        'Controlled geometry: By using particles with precisely known dimensions and shapes, we isolate '
        'the effect of geometry on settling dynamics, free from confounding factors present in environmentally '
        'weathered samples.',

        'Statistical rigor: With up to 3 repeats per particle, we quantify the inherent variability in '
        'settling behavior, providing uncertainty bounds that single-measurement studies cannot offer.',

        'Complementary size range: Our particles (Deq = 1.84–9.25 mm) extend the size range beyond '
        'Lofty\'s environmental MPs (Deq = 0.40–1.86 mm), together covering a broad spectrum of microplastic '
        'sizes.',

        'Machine learning prediction: We demonstrate that settling velocity can be predicted with R² = 0.859 '
        'using only geometric features — a capability absent from Lofty et al. and most existing 3D settling '
        'studies.',
    ]

    doc.add_paragraph('Key methodological advantages of this study:')
    for adv in advantages:
        doc.add_paragraph(adv, style='List Bullet')

    doc.add_page_break()

    # ========== 4. KEY FINDINGS ==========
    add_heading_styled(doc, '4. Key Findings', level=1)

    key_findings = [
        ('3D trajectories reveal shape-dependent settling dynamics',
         f'Of the 293 successfully reconstructed 3D trajectories, tortuosity ranged from '
         f'{df.tortuosity_pct.min():.1f}% to {df.tortuosity_pct.max():.1f}%. Elliptic cylinders '
         f'exhibited the highest mean tortuosity (22.9%), nearly three times that of cylindrical '
         f'particles (7.7%), confirming that particle flatness drives non-linear settling behavior.'),

        ('Lateral drift is significant for non-spherical particles',
         f'Drift gradients up to {df.drift_gradient.max():.3f} were observed, meaning some particles '
         f'traveled laterally up to 36% of their vertical descent distance. Horizontal drift velocities '
         f'averaged {df.horizontal_drift_velocity_cm_s.mean():.1f} cm/s, representing '
         f'{(df.horizontal_drift_velocity_cm_s.mean() / df.settling_velocity_cm_s.mean() * 100):.0f}% '
         f'of the mean settling velocity.'),

        ('Particle geometry alone predicts settling velocity',
         'The ensemble ML model achieves R² = 0.859 using only 9 geometric features. Adding 3D trajectory '
         'metrics (drift, tortuosity) does not improve prediction, indicating that settling dynamics are a '
         'consequence of geometry, not independent information.'),

        ('Controlled experiments complement environmental studies',
         'Our controlled particles (Deq = 1.84–9.25 mm, known density) extend and validate the findings '
         'of Lofty et al. (2026) obtained from environmental MPs (Deq = 0.40–1.86 mm, unknown density). '
         'Together, the two studies span a wide range of particle sizes and Reynolds numbers.'),
    ]

    for i, (title, desc) in enumerate(key_findings, 1):
        p = doc.add_paragraph()
        run = p.add_run(f'Finding {i}: {title}')
        run.bold = True
        run.font.size = Pt(11)
        doc.add_paragraph(desc)
        doc.add_paragraph()

    # ========== 5. CONCLUSIONS ==========
    add_heading_styled(doc, '5. Conclusions', level=1)

    doc.add_paragraph(
        'This study presents a comprehensive investigation of the three-dimensional settling dynamics '
        'of microplastic particles with controlled geometries. By combining dual-camera 3D trajectory '
        'reconstruction with machine learning-based velocity prediction, we provide new insights into '
        'how particle shape governs settling behavior in quiescent water.'
    )

    doc.add_paragraph(
        'Our results demonstrate that particle shape has a profound effect on settling dynamics beyond '
        'the vertical settling velocity. Flat, elongated particles (elliptic cylinders) follow highly '
        'tortuous paths with significant lateral displacement, while compact shapes (cubes, cylinders) '
        'settle along more linear trajectories. These findings have direct implications for modeling '
        'microplastic transport in rivers and marine environments, where assumptions of linear settling '
        'may significantly underestimate the lateral dispersion of non-spherical particles.'
    )

    doc.add_paragraph(
        'The machine learning model demonstrates that settling velocity can be reliably predicted '
        '(R² = 0.859) from particle geometry alone, without requiring trajectory measurements. This '
        'provides a practical tool for estimating settling behavior from easily measurable particle '
        'properties. The finding that 3D trajectory features do not improve prediction further '
        'supports the conclusion that settling dynamics are a direct consequence of particle geometry '
        'and density.'
    )

    doc.add_paragraph(
        'Future work should extend this analysis to smaller particle sizes (< 1 mm), include buoyant '
        '(rising) particles, and investigate settling behavior under turbulent conditions to better '
        'represent real environmental scenarios.'
    )

    # ========== SAVE ==========
    out_path = os.path.join(BASE_DIR, '3D_Settling_Dynamics_Report.docx')
    doc.save(out_path)
    print(f'Report saved to: {out_path}')
    return out_path


if __name__ == '__main__':
    generate_report()
