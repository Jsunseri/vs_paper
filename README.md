- here I'll try to describe a general workflow, identifying the relevant
  scripts and any assumptions they make about associated inputs: 

  > creating a summary file with sdsorter (and pasting columns from accompanying CSVs) 
    + use make_sdsorter_summary.sh; takes path to top-level directory, assumes
      docked files are in that directory and named *_docked.sdf.gz,
      looks for RFScore outputs as CSVs in the same directory, 
      looks for CNN output in subdirectories relative to the first directory
      and named after the model

    + creates output space-separated CSV with column names in first row, i.e. 
Rank Title Vina MW Target File RFScore-VS RFScore-4	dense_seed0_CNNaffinity dense_seed0_CNNscore dense_seed1_CNNaffinity dense_seed1_CNNscore dense_seed2_CNNaffinity dense_seed2_CNNscore dense_seed3_CNNaffinity dense_seed3_CNNscore dense_seed4_CNNaffinity dense_seed4_CNNscore crossdock_default2018_seed0_CNNaffinity crossdock_default2018_seed0_CNNscore crossdock_default2018_seed1_CNNaffinity crossdock_default2018_seed1_CNNscore crossdock_default2018_seed2_CNNaffinity crossdock_default2018_seed2_CNNscore crossdock_default2018_seed3_CNNaffinity crossdock_default2018_seed3_CNNscore crossdock_default2018_seed4_CNNaffinity crossdock_default2018_seed4_CNNscore general_default2018_seed0_CNNaffinity general_default2018_seed0_CNNscore general_default2018_seed1_CNNaffinity general_default2018_seed1_CNNscore general_default2018_seed2_CNNaffinity general_default2018_seed2_CNNscore general_default2018_seed3_CNNaffinity general_default2018_seed3_CNNscore general_default2018_seed4_CNNaffinity general_default2018_seed4_CNNscore 

    + subsequent steps take this output, which can be easily generated from
    whatever your score files are by just taking the sdsorter output and
    pasting any additional columns together

  > creating a pickle of a Pandas DataFrame of that info 

    + use sdsorter_to_pickle.py; takes list of dirnames that contain files
    named "sdsorter.summary" that have format as described in step 1 above,
    dumps out a pickle of the Pandas DataFrame created from that information
    (negating the Vina column and assigning a label based on the filename)

  > creating a summary file of predictions, typically per-compound 

    + make_summary.py creates output CSV files with columns that are 
    LABEL PREDICTION TARGET TITLE METHOD, and the prediction is obtained by taking
    the maximum score after doing a groupby with respect to target and compound

    + summarize_cnn_alltrain_predictions.py is an old version of this type of
    script that just does this for the CNN models, probably deprecated but
    keeping around just in case
 
  > calculating summary data
    
    + aucbytarget.py creates by-target grids of plots showing ROC curves for
    all methods whose ${methodname}.summary files are provided, as well as
    boxplots of AUCs

    + early_enrichment.py creates by-target barplots and by-method boxplots of
    early enrichment and normalized early enrichment; the user can specify the
    ratio but it defaults to 1%

    + auc_stripplot.py creates by-target stripplots of AUC, was basically a
    one-off that I didn't end up using
 
    + correlation_summary.py computes correlations between predictions for
    different methods, over all compounds as well as conditioned on class; it
    also computes and plots the intersection between the highest-ranked
    compounds between different methods

    + scores_vs_poses.py does the grid of plots showing score distribution and
    correlation per class between all methods

    + pose_variance_sensitivity.py calculates AUC and EF1% when dropping
    percentiles based on variance as well as mean variance in each quartile

  > DeepChem scripts

    + train_atomicconv.py should in theory train a DeepChem 3D Atomic
    Convolution model, but I've been having problems with it so it's a WIP

  > scripts for simple descriptor models

    + train_simpledescriptor_models.py fits a set of available models from
    sklearn to input data; it can also do cross validation (with types files
    provided as folds or doing a random split) and hyperparameter optimization

    + predict_simpledescriptor_models.py uses trained simple descriptor models,
    provided as joblib files, to make predictions for an input types file

    + dump_fingerprints.py dumps a CSV of the fingerprints for the molecules in
    an input types file

    + make_extra_summary_features.py dumps a CSV of additional features that
    match the order of a types file, used e.g. for matching the CNN predictions
    to the types files used to train/test simple descriptor models

    + make_moltitle_exclusionlist.py will output the list of file+molname
    instances B\A, given A=summaryfile and B=typesfile

  > figure style-related scripts

    + vspaper.mpltstyle is a matplotlib stylesheet that modifies a few things
    from the seaborn-white parameters

    + vspaper_settings.py includes palettes, publication-ready names for
    methods, additional info that is sometimes included with targets in
    figures, markers chosen for swarmplots, and it also loads the stylesheet
    when it's included

  > accessory scripts for other tasks

    + extract_pdbbind_data.py generate summary data from a PDBBind index file

    + vsplot_summary_fordavid.py compares Vina and a user-provided CNN model
    (for the submission I used the Dense CNNaffinity model) with AUC boxplots
    on DUD-E and LIT-PCBA
