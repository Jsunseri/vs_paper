import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

backup_palette = ['#003f5c', '#665191', '#d45087', '#f95d6a', '#ffa600',
        '#2f4b7c', '#a05195', '#ff7c43']

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
paper_palettes['Vinardo'] = '#F2CC86'
paper_palettes['dense-CNNscore-mean'] = '#60DFC4'
paper_palettes['dense-CNNaffinity-mean'] = '#5dc3bf'
paper_palettes['dense_consensus'] = '#cdf2dd'
paper_palettes['crossdock_default2018-CNNscore-mean'] = '#FFA502'
paper_palettes['crossdock_default2018-CNNaffinity-mean'] = '#C96424'
paper_palettes['crossdock_default2018_consensus'] = '#f0c4a7'
paper_palettes['general_default2018-CNNscore-mean'] = '#B404F3'
paper_palettes['general_default2018-CNNaffinity-mean'] = '#2A11A0'
paper_palettes['general_default2018_consensus'] = '#e1d2e9'
paper_palettes['rf-score-vs'] = '#7ACFE5'
paper_palettes['rf-score-4'] = '#024B78'
paper_palettes['DUD-E\ndescriptors,\nGeneral Set fit,\n'] = '#D4AD00'
paper_palettes['MUV\ndescriptors,\nGeneral Set fit,\n'] = '#E7E000'

paper_palettes['CNN'] = paper_palettes['CNN Scoring Rescore']
paper_palettes['CNNscore'] = paper_palettes['CNN Scoring Rescore']
paper_palettes['CNNaffinity'] = paper_palettes['CNN Affinity Rescore']
paper_palettes['Dense\n(Pose)'] = paper_palettes['dense-CNNscore-mean']
paper_palettes['Dense\n(Affinity)'] = paper_palettes['dense-CNNaffinity-mean']
paper_palettes['dense-aff-mean'] = paper_palettes['dense-CNNaffinity-mean']
paper_palettes['Dense\n(Consensus)'] = paper_palettes['dense_consensus']
paper_palettes['Cross-Docked\n(Pose)'] = paper_palettes['crossdock_default2018-CNNscore-mean']
paper_palettes['crossdocked2018-CNNaffinity-mean'] = paper_palettes['crossdock_default2018-CNNaffinity-mean']
paper_palettes['Cross-Docked\n(Affinity)'] = paper_palettes['crossdock_default2018-CNNaffinity-mean']
paper_palettes['Cross-Docked\n(Consensus)'] = paper_palettes['crossdock_default2018_consensus']
paper_palettes['General\n(Pose)'] = paper_palettes['general_default2018-CNNscore-mean']
paper_palettes['General\n(Affinity)'] = paper_palettes['general_default2018-CNNaffinity-mean']
paper_palettes['generalset2018-CNNaffinity-mean'] = paper_palettes['general_default2018-CNNaffinity-mean']
paper_palettes['General\n(Consensus)'] = paper_palettes['general_default2018_consensus']
paper_palettes['Dense (Pose)'] = paper_palettes['dense-CNNscore-mean']
paper_palettes['Dense (Affinity)'] = paper_palettes['dense-CNNaffinity-mean']
paper_palettes['Cross-Docked\n(Pose)'] = paper_palettes['crossdock_default2018-CNNscore-mean']
paper_palettes['Cross-Docked\n(Affinity)'] = paper_palettes['crossdock_default2018-CNNaffinity-mean']
paper_palettes['General (Pose)'] = paper_palettes['general_default2018-CNNscore-mean']
paper_palettes['General (Affinity)'] = paper_palettes['general_default2018-CNNaffinity-mean']
paper_palettes['RFScore-VS'] = paper_palettes['rf-score-vs']
paper_palettes['RFScore-4'] = paper_palettes['rf-score-4']
paper_palettes['RF DUD-E'] = backup_palette[2]
paper_palettes['RF MUV'] = backup_palette[3]
paper_palettes['RF Refined\n(DUD-E features)'] = paper_palettes['RF DUD-E'] 
paper_palettes['RF Refined\n(MUV features)'] = paper_palettes['RF MUV']

name_map = {'dense-CNNscore-mean': 'Dense\n(Pose)', 'dense-CNNaffinity-mean': 'Dense\n(Affinity)',
        'dense-CNNscore_CNNaffinity-poselevel_product': 'Dense\nPose %s Affinity' %'$\u00D7$',

        'crossdocked_default2018-CNNscore-mean': 'Cross-Docked\n(Pose)', 
        'crossdock_default2018-CNNscore-mean': 'Cross-Docked\n(Pose)', 
        'crossdock_default2018-CNNaffinity-mean': 'Cross-Docked\n(Affinity)', 
        'crossdock_default2018-CNNscore_CNNaffinity-poselevel_product':
        'Cross-Docked\nPose %s Affinity' %'$\u00D7$',

        'general_default2018-CNNscore-mean': 'General\n(Pose)',
        'general_default2018-CNNaffinity-mean': 'General\n(Affinity)', 
        'general_default2018-CNNscore_CNNaffinity-poselevel_product':
        'General\nPose %s Affinity' %'$\u00D7$',

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

        'Vina': 'Vina',
        'dense': 'Dense', 
        'crossdock_default2018': 'Cross-Docked', 
        'general_default2018': 'General',
        'CNNscore': 'Pose',
        'CNNaffinity': 'Affinity'
        }

reverse_map = {'Dense\n(Pose)': 'dense-CNNscore-mean',
        'Dense\n(Affinity)': 'dense-CNNaffinity-mean', 
        'Dense\nPose %s Affinity' %'$\u00D7$': 'dense-CNNscore_CNNaffinity-poselevel_product',

        'Cross-Docked\n(Pose)': 'crossdock_default2018-CNNscore-mean', 
        'Cross-Docked\n(Affinity)': 'crossdock_default2018-CNNaffinity-mean',
        'Cross-Docked\nPose %s Affinity' %'$\u00D7$':
        'crossdock_default2018-CNNscore_CNNaffinity-poselevel_product',

        'General\n(Pose)': 'general_default2018-CNNscore-mean', 
        'General\n(Affinity)': 'general_default2018-CNNaffinity-mean', 
        'General\nPose %s Affinity' %'$\u00D7$':
        'general_default2018-CNNscore_CNNaffinity-poselevel_product', 

        'RFScore-VS': 'rfscore-vs', 
        'RFScore-4': 'rf-score-4', 

        'Dense\n(Consensus)': 'dense_consensus', 
        'Cross-Docked\n(Consensus)': 'crossdock_default2018_consensus', 
        'General\n(Consensus)': 'general_default2018_consensus',

        'RF Refined\n(DUD-E features)': 'RF-Refined-DUDe', 
        'RF Refined\n(MUV features)': 'RF-Refined-MUV'
        }

for model in ['dense', 'crossdock_default2018', 'general_default2018']:
    for layer in ['CNNscore', 'CNNaffinity']:
        in_name = '%s-%smaxthen-mean-div-std' %(model,layer)
        out_name = '%s\n(%s)\n%s-Adjusted\nPrediction' %(name_map[model], name_map[layer], r'$\sigma$')
        name_map[in_name] = out_name
        reverse_map[out_name] = in_name

        in_name = '%s-%s-mean-div-expstd' %(model,layer)
        out_name = '%s\n(%s)\n%s-Adjusted\nPrediction' %(name_map[model], name_map[layer], r'$e^{\sigma}$')
        name_map[in_name] = out_name
        reverse_map[out_name] = in_name

        in_name = '%s-%s-mean-div-std' %(model,layer)
        out_name = '%s\n(%s)\n%s\n%s' %(name_map[model], name_map[layer], '$\u2014$', r'$\sigma$')
        name_map[in_name] = out_name
        reverse_map[out_name] = in_name

        in_name = '%s-%s-scoregap' %(model, layer)
        out_name = '%s\n(%s)\n%s\n(Max-Min)' %(name_map[model], name_map[layer], '$\u00D7$')
        name_map[in_name] = out_name
        reverse_map[out_name] = in_name

cnn_lookup = {
        'generalScore': 'General\n(Pose)',
        'generalAff': 'General\n(Affinity)',
        'generalScore_generalAff': 'General (Pose)\nand\nGeneral (Affinity)'
        }
simple_models = ['KNN', 'SVM', 'GBT', 'DT', 'RF', 'Lasso', 'GBT', 'Logistic']
for mname in simple_models:
    paper_palettes['DUD-E\ndescriptors,\nGeneral Set fit,\n%s' %mname] = \
                                paper_palettes['DUD-E\ndescriptors,\nGeneral Set fit,\n']
    paper_palettes['MUV\ndescriptors,\nGeneral Set fit,\n%s' %mname] = \
                                paper_palettes['MUV\ndescriptors,\nGeneral Set fit,\n']

    name_map['MUVdesc_general_%s' %mname] = 'MUV\ndescriptors,\nGeneral Set fit,\n%s' %mname
    reverse_map['MUV\ndescriptors,\nGeneral Set fit,\n%s' %mname] = 'MUVdesc_general_%s' %mname
    name_map['DUDEdesc_general_%s' %mname] = 'DUD-E\ndescriptors,\nGeneral Set fit,\n%s' %mname
    reverse_map['DUD-E\ndescriptors,\nGeneral Set fit,\n%s' %mname] = 'DUDEdesc_general_%s' %mname

    name_map['MUVdesc_LITPCBAfit_noCNN_%s' %mname] = 'MUV\ndescriptors,\nLIT-PCBA fit,\n%s' %mname
    reverse_map['MUV\ndescriptors,\nLIT-PCBA fit,\n%s' %mname] = 'MUVdesc_LITPCBAfit_noCNN_%s' %mname
    name_map['DUDEdesc_LITPCBAfit_noCNN_%s' %mname] = 'DUD-E\ndescriptors,\nLIT-PCBA fit,\n%s' %mname
    reverse_map['DUD-E\ndescriptors,\nLIT-PCBA fit,\n%s' %mname] = 'DUDEdesc_LITPCBAfit_noCNN_%s' %mname
    name_map['MUVdesc_DUDEfit_noCNN_%s' %mname] = 'MUV\ndescriptors,\nDUD-E fit,\n%s' %mname
    reverse_map['MUV\ndescriptors,\nDUD-E fit,\n%s' %mname] = 'MUVdesc_DUDEfit_noCNN_%s' %mname
    name_map['DUDEdesc_DUDEfit_noCNN_%s' %mname] = 'DUD-E\ndescriptors,\nDUD-E fit,\n%s' %mname
    reverse_map['DUD-E\ndescriptors,\nDUD-E fit,\n%s' %mname] = 'DUDEdesc_DUDEfit_noCNN_%s' %mname

    name_map['MUVdesc_fingerprint_general_%s' %mname] = \
                              'MUV\ndescriptors\nwith ECFP4 fingerprint\nGeneral Set fit,\n%s' %mname
    reverse_map['MUV\ndescriptors\nwith ECFP4 fingerprint\nGeneral Set fit,\n%s' %mname] = \
                              'MUVdesc_fingerprint_general_%s' %mname              
    name_map['DUDEdesc_fingerprint_general_%s' %mname] = \
                              'DUD-E\ndescriptors\nwith ECFP4 fingerprint\nGeneral Set fit,\n%s' %mname
    reverse_map['DUD-E\ndescriptors\nwith ECFP4 fingerprint\nGeneral Set fit,\n%s' %mname] = \
                              'DUDEdesc_fingerprint_general_%s' %mname              
    for cnn in ['generalScore', 'generalAff', 'generalScore_generalAff']:
        cnn_name = cnn_lookup[cnn]
        name_map['MUVdesc_%s_%s' %(cnn,mname)] = 'MUV\ndescriptors\nwith\n%s,\n%s' %(cnn_name,mname)
        reverse_map['MUV\ndescriptors\nwith\n%s,\n%s' %(cnn_name,mname)] = 'MUVdesc_%s_%s' %(cnn,mname)
        name_map['DUDEdesc_%s_%s' %(cnn,mname)] = 'DUD-E\ndescriptors\nwith\n%s,\n%s' %(cnn_name,mname)
        reverse_map['DUD-E\ndescriptors\nwith\n%s,\n%s' %(cnn_name,mname)] = 'DUDEdesc_%s_%s' %(cnn,mname)

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

litpcba_order = [
        'ADRB2',
        'ALDH1', 
        'ESR1 ago', 
        'ESR1 ant', 
        'FEN1', 
        'GBA', 
        'IDH1', 
        'KAT2A', 
        'MAPK1', 
        'MTORC1', 
        'OPRK1', 
        'PKM2', 
        'PPARG', 
        'TP53', 
        'VDR'
        ]

litpcba_ntemplates = {
        'ADRB2': 8, 
        'ALDH1': 8, 
        'ESR1 ago': 15, 
        'ESR1_ago': 15, 
        'ESR1 ant': 15, 
        'ESR1_ant': 15, 
        'FEN1': 1, 
        'GBA': 6, 
        'IDH1': 14, 
        'KAT2A': 3, 
        'MAPK1': 15, 
        'MTORC1': 11, 
        'OPRK1': 1, 
        'PKM2': 9, 
        'PPARG': 15, 
        'TP53': 6, 
        'VDR': 2
        }

# u2665 == filled heart; u2602 == umbrella; u260e == telephone; 
# u260f == telephone2;
# u265b == chessqueen; u265e == chess knight; u265C == rook;
# u266A == eighth note; u1f71b == silver; u231a == watch;
# u263E == last quarter moon; u232C == benzene; u2328 == keyboard;
# u2740 == fleuron1; u2702 == scissors; u270E == pencil1; u2767 == fleuron2;
# u2710 == pencil2; u27b7 == arrow; u2708 == airplane; u274[4-6] == snowflakes;
# u266E == naturalsign; u222F == surface integral; u269C == fleur-de-lys;
# u29AC == angle; u27BF == [undefined but i like the dummy symbol]; u22CA == semijoin; 
swarm_markers = [r'$\clubsuit$', r'$\spadesuit$', '$\u2665$', '*', 
        '$\u265B$', '$\u265E$', '$\u265C$',
        '$\u2602$', '$\u266A$', '$\u2702$', 
        '$\u2708$', '$\u2744$', '$\u27b7$', 
        '$\u29AC$', '$\u222F$', '$\u263E$', 
        '$\u22CA$', 
        '$\u266E$', '$\u27BF$', 
        '^', '>', 's', 'o', '<', 'X', 'v', 'h', 
        '+', '1', '2', '3', '4', 'x']

marker_sizes = {
        '$\u2702$': 16,
        '$\u265B$': 16
        }

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
