import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

backup_palette = sns.color_palette("hls", n_colors=10, desat=.5).as_hex()

paper_palettes = {}
paper_palettes['Vina'] = '#000000' #the original CNN paper used ccbb44
paper_palettes['CSAR'] = '#332288'
paper_palettes['DUD-E'] = '#4477aa'
paper_palettes['2:1'] = '#88ccee'
paper_palettes['CNN Affinity Rescore'] = '#6da4c0'
paper_palettes['CNN Affinity Refine'] = '#332288'
paper_palettes['CNN Scoring Rescore'] = '#ffd91c'
paper_palettes['CNN Scoring Refine'] = '#877b25'
paper_palettes['Experiment'] = '#498540'
paper_palettes['CNN'] = '#ffd91c'
paper_palettes['CNNscore'] = '#ffd91c'
paper_palettes['CNNaffinity'] = '#6da4c0'
paper_palettes['Vinardo'] = '#BDC3C7'
paper_palettes['dense-CNNscore-mean'] = '#82E0AA'
paper_palettes['Dense\n(Pose)'] = '#82E0AA'
paper_palettes['dense-CNNaffinity-mean'] = '#28B463'
paper_palettes['Dense\n(Affinity)'] = '#28B463'
paper_palettes['dense-aff-mean'] = '#28B463'
paper_palettes['dense_consensus'] = '#cdf2dd'
paper_palettes['Dense\n(Consensus)'] = '#cdf2dd'
paper_palettes['crossdock_default2018-CNNscore-mean'] = '#E59866'
paper_palettes['Cross-Docked\n(Pose)'] = '#E59866'
paper_palettes['crossdock_default2018-CNNaffinity-mean'] = '#BA4A00'
paper_palettes['crossdocked2018-CNNaffinity-mean'] = '#BA4A00'
paper_palettes['Cross-Docked\n(Affinity)'] = '#BA4A00'
paper_palettes['crossdock_default2018_consensus'] = '#f0c4a7'
paper_palettes['Cross-Docked\n(Consensus)'] = '#f0c4a7'
paper_palettes['general_default2018-CNNscore-mean'] = '#b788cb'
paper_palettes['General\n(Pose)'] = '#b788cb'
paper_palettes['general_default2018-CNNaffinity-mean'] = '#9B59B6'
paper_palettes['General\n(Affinity)'] = '#9B59B6'
paper_palettes['generalset2018-CNNaffinity-mean'] = '#9B59B6'
paper_palettes['general_default2018_consensus'] = '#e1d2e9'
paper_palettes['General\n(Consensus)'] = '#e1d2e9'
paper_palettes['rf-score-vs'] = '#D98880'
paper_palettes['rf-score-4'] = '#A93226'
paper_palettes['Dense (Pose)'] = '#82E0AA'
paper_palettes['Dense (Affinity)'] = '#28B463'
paper_palettes['Cross-Docked\n(Pose)'] = '#E59866'
paper_palettes['Cross-Docked\n(Affinity)'] = '#BA4A00'
paper_palettes['General (Pose)'] = '#b788cb'
paper_palettes['General (Affinity)'] = '#9B59B6'
paper_palettes['RFScore-VS'] = '#5DADE2'
paper_palettes['RFScore-4'] = '#2874A6'
paper_palettes['RF DUD-E'] = backup_palette[2]
paper_palettes['RF MUV'] = backup_palette[3]
paper_palettes['RF Refined\n(DUD-E features)'] = backup_palette[2]
paper_palettes['RF Refined\n(MUV features)'] = backup_palette[3]

name_map = {'dense-CNNscore-mean': 'Dense\n(Pose)', 'dense-CNNaffinity-mean': 'Dense\n(Affinity)',
        'crossdocked_default2018-CNNscore-mean': 'Cross-Docked\n(Pose)', 
        'crossdock_default2018-CNNscore-mean': 'Cross-Docked\n(Pose)', 
        'crossdock_default2018-CNNaffinity-mean': 'Cross-Docked\n(Affinity)', 
        'general_default2018-CNNscore-mean': 'General\n(Pose)',
        'general_default2018-CNNaffinity-mean': 'General\n(Affinity)', 
        'rfscore-vs': 'RFScore-VS',
        'rf-score-4': 'RFScore-4',
        'dense-aff-mean': 'Dense\n(Affinity)',
        'crossdocked2018-CNNaffinity-mean': 'Cross-Docked\n(Affinity)', 
        'generalset2018-CNNaffinity-mean': 'General\n(Affinity)', 
        'dense_consensus': 'Dense\n(Consensus)', 
        'crossdock_default2018_consensus': 'Cross-Docked\n(Consensus)', 
        'general_default2018_consensus': 'General\n(Consensus)',
        'RF-Refined-DUDe': 'RF Refined\n(DUD-E features)',
        'RF-Refined-MUV': 'RF Refined\n(MUV features)', 
        'MUVdesc_general_DT': 'MUV\ndescriptors,\nGeneral Set fit,\nDT',
        'MUVdesc_general_GBT': 'MUV\ndescriptors,\nGeneral Set fit,\nGBT'
        }

reverse_map = {'Dense\n(Pose)': 'dense-CNNscore-mean',
        'Dense\n(Affinity)': 'dense-CNNaffinity-mean', 
        'Cross-Docked\n(Pose)': 'crossdock_default2018-CNNscore-mean', 
        'Cross-Docked\n(Affinity)': 'crossdock_default2018-CNNaffinity-mean',
        'General\n(Pose)': 'general_default2018-CNNscore-mean', 
        'General\n(Affinity)': 'general_default2018-CNNaffinity-mean', 
        'RFScore-VS': 'rfscore-vs', 
        'RFScore-4': 'rf-score-4', 
        'Dense\n(Consensus)': 'dense_consensus', 
        'Cross-Docked\n(Consensus)': 'crossdock_default2018_consensus', 
        'General\n(Consensus)': 'general_default2018_consensus',
        'RF Refined\n(DUD-E features)': 'RF-Refined-DUDe', 
        'RF Refined\n(MUV features)': 'RF-Refined-MUV'
        }

litpcba_successes = {
        'ADRB2': ['2D', '3D', 'SD'],
        'ALDH1': ['2D'],
        'ESR1 ago': ['2D', '3D'],
        'ESR1_ago': ['2D', '3D'],
        'ESR1 ant': ['3D'],
        'ESR1_ant': ['3D'],
        'FEN1': ['SD'],
        'GBA': ['2D', 'SD'],
        'IDH1': ['2D'],
        'KAT2A': ['SD'],
        'MAPK1': ['3D'],
        'MTORC1': ['3D'],
        'OPRK1': ['2D', 'SD'],
        'PKM2': ['2D', '3D'],
        'PPARG': ['2D', '3D', 'SD'],
        'TP53': ['3D'],
        'VDR': ['2D'],
        }

swarm_markers = [r'$\clubsuit$', r'$\spadesuit$', '^', '>', '*',
        's', 'o', '<', 'X', 'v', 'h', r'$\P$', '+', '1', '2', '3', '4', 'x']

blue_star = mlines.Line2D([], [], color='black', marker='^', linestyle='None',
                                  markersize=10, label='Blue stars')
red_square = mlines.Line2D([], [], color='red', marker='s', linestyle='None',
                                  markersize=10, label='Red squares')
purple_triangle = mlines.Line2D([], [], color='purple', marker='^',
        linestyle='None',
                                  markersize=10, label='Purple triangles')

dirname = os.path.dirname(os.path.realpath(__file__))
sheetname = 'vspaper.mpltstyle'
plt.style.use(os.path.join(dirname, sheetname))
