# LING-L715
Hate Speech on Levantine Tweets

Datasets:
    1. DART

        a.) /cf-data 
                Contains .txt files, each with Tweets from that specific dialect
                (e.g. EGY.txt contains Tweets of Egyptain dialect)
                Format: 3 columns; (1) score, (2) tweet_ID, (3) tweet_text

        b.) get_DART_transcripts.py
                Code to extract Tweets from .txt files in /cf-data
                Creates 2 new files: (1) clean_LEV.tsv, (2) clean_NONLEV.tsv
                The --other_paths argument takes at least one path; simply write out the paths one after another
                From terminal: python get_DART_transcripts.py --lev_path [PATH TO '/DART/cf-data/LEV.txt'] --other_paths [PATHS TO '/DART/cf-data/[^LEV].txt']

        c.) clean_LEV.tsv
                One of the output .tsv files after running get_DART_transcripts.py
                Contains Levantine-only Tweets, each labeled with "LEV"

        d.) clean_NONLEV.tsv
                One of the output .tsv files after running get_DART_transcripts.py
                Contains Levantine-only Tweets, each labeled with "NONLEV"

        e.) classify_LEV.py
                From terminal: python classify_LEV.py --clean_lev_path [path to 'clean_LEV.tsv'] --clean_nonlev_path [PATH TO 'clean_NONLEV.tsv']

        f.) classify_LEV_metrics.txt
                Results from Linear SVM classifier using SelectKBest(chi2, k=500)
                Binary classification ("LEV" vs "NONLEV")
        
        g.) 
                

    2. LHSAB
    3. OSACT