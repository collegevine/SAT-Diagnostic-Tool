import pandas as pd
from flask_table import Table, Col
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import random



data_assets = {
    'math1_ans' : './data/Data Assets - Math1-ans.csv',
    'math2_ans' : './data//Data Assets - Math2-ans.csv',
    'math_chart' : './data/Data Assets - Math-score-chart.csv',
    'verbal_ans' : './data/Data Assets - Verbal-ans.csv',
    'writing_ans' : './data/Data Assets - Writing-ans.csv',
    'verbal_scale' : './data/Data Assets - Verbal-scaled-score.csv',
    'verbal_score' : './data/Data Assets - Verbal-score-chart.csv',
    'combined_score' : './data/Data Assets - Combined-percentile.csv'
}

def run_analysis(ans_dict):
    total_scale_df = pd.read_csv(data_assets['combined_score'])
    m = calculate_math_score(ans_dict)
    v = calculate_verbal_score(ans_dict)
    m.update(v)
    total_score = m.get('math_score') + v.get('verbal_score')
    m['total_score'] = total_score
    m['total_percentile'] = total_scale_df.loc[total_scale_df['score'] == total_score ]['percentile'].tolist()[0]
    return m

def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z

def lookup_ans(ans_dict, idx, keyword):
    key = f'{keyword}_{idx}'
    return ans_dict.get(key,"")

def fmt_percentage(num,denom):
    x = (num / denom) * 100
    return "{0:.2f}".format(x)

def fmt_improve(i):
    if isinstance(i,float):
        i = i / 10
        i = int(round(i,0))
        i = i * 10
        return str(i)
    return i

def agg_counts_dict(df):
    return dict(df.apply(pd.value_counts).fillna(0).apply(sum, axis=1))

def eval_str(istr):
    estr = eval(istr)
    if isinstance(estr, float):
        return str(round(estr, 2))
    else:
        return istr

def qeq(query, ans):
    if query is '':
        return False
    if query in ['A','B','C','D']:
        return query == ans
    elif 'or' in ans:
        anslist = ans.split(' or ')
        return any([eval_str(query) == x for x in ans.split(' or ')])
    else:
        return eval_str(query) == ans

def make_file():
    x = str(random.randint(500,1000000))
    return f'./img/{x}.png'


def plot_math(dicts):
    df = pd.DataFrame.from_dict(dicts.get('math_difficulty'))
    df = df.sort_values(['difficulty'])

    objects = ('Level 1 - Easy', 'Level 2 - Easy to Medium', 'Level 3 - Medium', 'Level 4 - Hard', 'Level 5 - Super-hard')
    y_pos = np.arange(len(objects))
    num_wrong = list(df['wrong'])

    plt.bar(y_pos, num_wrong, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Number of Questions')
    plt.xlabel('Difficulty Level of Question')
    plt.title('SAT Math Questions Wrong By Difficulty')
    plt.xticks(y_pos, objects, rotation='vertical')
    ffile = make_file()
    plt.savefig(ffile, bbox_inches='tight')
    plt.close()
    return ffile


def plot_verbal(dicts):
    v = pd.DataFrame.from_dict(dicts.get('reading_difficulty'))[['difficulty', 'wrong']]
    w = pd.DataFrame.from_dict(dicts.get('writing_difficulty'))[['difficulty', 'wrong']]
    m = pd.merge(v,w, on='difficulty')
    m['wrong'] = m.apply(lambda row: row['wrong_x'] + row['wrong_y'],axis=1)
    df = m.sort_values(['difficulty'])

    objects = ('Level 1 - Easy', 'Level 2 - Easy to Medium', 'Level 3 - Medium', 'Level 4 - Hard', 'Level 5 - Super-hard')
    ## NOTE there is no LEVEL 1 in VERBAL section
    y_pos = np.arange(len(objects))
    num_wrong = [0] + list(df['wrong'])

    plt.bar(y_pos, num_wrong, align='center')
    plt.xticks(y_pos, objects, rotation='vertical')
    plt.ylabel('Number of Questions')
    plt.xlabel('Difficulty Level of Question')
    plt.title('SAT Verbal Questions Wrong By Difficulty')
    ffile = make_file()
    plt.savefig(ffile, bbox_inches='tight')
    plt.close()
    return ffile

def mk_concept_dict(miss, total):
    llist = [{'concept': k, 'wrong': int(miss.get(k,0)), 'pct' : fmt_percentage(miss.get(k,0), sum(total.values())), }for k in total.keys()]
    newlist = sorted(llist, key=lambda k: float(k['pct']), reverse=True)
    return newlist

def mk_diff_dict(miss, total):
    return [{'difficulty': k, 'wrong': miss.get(k,0), 'pct' : fmt_percentage(miss.get(k,0), sum(total.values())) }  for k in total.keys()]

def mk_explain_dict(df,section):
    df['section'] = section
    return df[['section', 'question', 'response', 'answer','explain']].to_dict('records')

def mk_improve_dict(miss, total):
    print(miss)
    print(total)
    total_N = float(sum(total.values()))
    llist = [{'concept': k, 'improvement': fmt_improve((float(miss.get(k,0))/float(total.get(k,0))) * (float(total.get(k,0))/total_N) * 800) } for k in total.keys() ]
    newlist = sorted(llist, key = lambda k: float(k['improvement']), reverse=True)
    print(llist)
    return newlist


def calculate_math_score(ans_dict):
    # MATH 1 Correct answers
    m1_ans_df = pd.read_csv(data_assets.get('math1_ans'))
    m1_ans_df['correct'] = m1_ans_df.apply(lambda row: qeq(str(lookup_ans(ans_dict, int(row['question']), 'math1')), row['answer']), axis=1)
    m1_ans_df['response'] = m1_ans_df.apply(lambda row: str(lookup_ans(ans_dict, int(row['question']), 'math1')), axis=1)
    m1_num_correct = sum(m1_ans_df['correct'])

    # Math 2 correct ansers
    m2_ans_df = pd.read_csv(data_assets.get('math2_ans'))
    m2_ans_df['correct'] = m2_ans_df.apply(lambda row: qeq(str(lookup_ans(ans_dict, int(row['question']), 'math2')), row['answer']), axis=1)
    m2_ans_df['response'] = m2_ans_df.apply(lambda row: str(lookup_ans(ans_dict, int(row['question']), 'math2')), axis=1)
    m2_num_correct = sum(m2_ans_df['correct'])

    # Math (Comb) Score & Percentile
    m_score_df = pd.read_csv(data_assets.get('math_chart'))
    total_correct = m1_num_correct + m2_num_correct
    score = int(m_score_df.loc[m_score_df['correct_ans'] == total_correct]['score'].tolist()[0])
    percentile = int(m_score_df.loc[m_score_df['correct_ans'] == total_correct]['percentile'].tolist()[0])

    # Match combined df
    m_ans_df = pd.concat([m1_ans_df, m2_ans_df])

    # Math Combined Incorrect Explainations
    #m_expalin_dict = mk_explain_dict(m_ans_df.loc[m_ans_df['correct'] == False], 'Math')

    # Math concepts
    m_total_concepts = agg_counts_dict(m_ans_df[['concept','concept2']])
    m_missed_concepts = agg_counts_dict(m_ans_df.loc[m_ans_df['correct'] == False][['concept','concept2']])
    m_concept_dict =  mk_concept_dict(m_missed_concepts, m_total_concepts)
    m_improve_dict = mk_improve_dict(m_missed_concepts, m_total_concepts)

    # Math Difficulty
    m_missed_diff = dict(m_ans_df.loc[m_ans_df['correct'] == False][['difficulty']].apply(pd.value_counts)['difficulty'])
    m_total_diff = agg_counts_dict(m_ans_df[['difficulty']])
    m_diff_dict = mk_diff_dict(m_missed_diff, m_total_diff)

    odict = {
        'math_score': score,
        'math_percentile': percentile,
        'math_concepts': m_concept_dict,
        'math_difficulty': m_diff_dict,
        #'math_explain' : m_expalin_dict
        'math_improve' : m_improve_dict
    }
    print("odict")
    return(odict)

def calculate_verbal_score(ans_dict):
    v_ans_df = pd.read_csv(data_assets.get('verbal_ans'))
    v_ans_df['correct'] = v_ans_df.apply(lambda row: lookup_ans(ans_dict, row['question'], 'verbal') == row['answer']  , axis=1)
    v_ans_df['response'] = v_ans_df.apply(lambda row: str(lookup_ans(ans_dict, int(row['question']), 'verbal')), axis=1)
    verbal_num_correct = sum(v_ans_df['correct'])

    w_ans_df = pd.read_csv(data_assets.get('writing_ans'))
    w_ans_df['correct'] = w_ans_df.apply(lambda row: lookup_ans(ans_dict, row['question'], 'writing') == row['answer']  , axis=1)
    w_ans_df['response'] = w_ans_df.apply(lambda row: str(lookup_ans(ans_dict, int(row['question']), 'writing')), axis=1)
    writing_num_correct = sum(w_ans_df['correct'])

    # V/W Combined Incorrect Explainations
    w_explain_dict = mk_explain_dict(w_ans_df.loc[w_ans_df['correct'] == False], 'Writing')
    v_explain_dict = mk_explain_dict(w_ans_df.loc[w_ans_df['correct'] == False], 'Reading')

    v_score_df = pd.read_csv(data_assets.get('verbal_score'))
    verbal_raw_score = v_score_df.loc[v_score_df['correct_ans'] == verbal_num_correct]['reading_raw_score'].tolist()[0]
    writing_raw_score = int(v_score_df.loc[v_score_df['correct_ans'] == writing_num_correct]['writing_raw_score'].tolist()[0])

    v_scale_df = pd.read_csv(data_assets.get('verbal_scale'))
    raw_score = verbal_raw_score + writing_raw_score

    score = int(v_scale_df.loc[v_scale_df['raw'] == raw_score ]['score'].tolist()[0])
    percentile = int(v_scale_df.loc[v_scale_df['raw'] == raw_score ]['percentile'].tolist()[0])

    # Verbal Concepts
    v_total_concepts = agg_counts_dict(v_ans_df[['concept','concept2']])
    v_missed_concepts = agg_counts_dict(v_ans_df.loc[v_ans_df['correct'] == False][['concept','concept2']])
    v_concept_dict =  mk_concept_dict(v_missed_concepts, v_total_concepts)

    # Writing Concepts
    w_total_concepts = agg_counts_dict(w_ans_df[['concept','concept2']])
    w_missed_concepts = agg_counts_dict(w_ans_df.loc[w_ans_df['correct'] == False][['concept','concept2']])
    w_concept_dict =  mk_concept_dict(w_missed_concepts, w_total_concepts)

    # Verbal Difficulty
    v_missed_diff = dict(v_ans_df.loc[v_ans_df['correct'] == False][['difficulty']].apply(pd.value_counts)['difficulty'])
    v_total_diff = agg_counts_dict(v_ans_df[['difficulty']])
    v_diff_dict = mk_diff_dict(v_missed_diff, v_total_diff)

    # Writing Difficulty
    w_missed_diff = dict(w_ans_df.loc[w_ans_df['correct'] == False][['difficulty']].apply(pd. value_counts)['difficulty'])
    w_total_diff = agg_counts_dict(w_ans_df[['difficulty']])
    w_diff_dict = mk_diff_dict(w_missed_diff, w_total_diff)


    # Verbal Section Improvement
    vw_total_concepts = merge_two_dicts(v_total_concepts, w_total_concepts)
    vw_missed_concepts = merge_two_dicts(v_missed_concepts, w_missed_concepts)
    vw_improve_dict = mk_improve_dict(vw_missed_concepts, vw_total_concepts)

    odict = {
        'verbal_score': score,
        'verbal_percentile': percentile,
        'reading_concepts': v_concept_dict,
        'reading_difficulty': v_diff_dict,
        'reading_explain' : v_explain_dict,
        'writing_concepts': w_concept_dict,
        'writing_difficulty': w_diff_dict,
        'writing_explain' : w_explain_dict,
        'verbal_improve' : vw_improve_dict
    }
    return(odict)


def format_profile_table(diff_dicts):
    table = ItemTable(diff_dicts)
    return(table)
