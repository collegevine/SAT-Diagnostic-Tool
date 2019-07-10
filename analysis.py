import pandas as pd
from flask_table import Table, Col
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import copy
from settings import APP_ROOT


data_assets = {
    'math1_ans' : './data/Data Assets - Math1-ans.csv',
    'math2_ans' : './data//Data Assets - Math2-ans.csv',
    'math_chart' : './data/Data Assets - Math-score-chart.csv',
    'verbal_ans' : './data/Data Assets - Verbal-ans.br.csv',
    'writing_ans' : './data/Data Assets - Writing-ans.br.csv',
    'verbal_scale' : './data/Data Assets - Verbal-scaled-score.csv',
    'verbal_score' : './data/Data Assets - Verbal-score-chart.csv',
    'combined_score' : './data/Data Assets - Combined-percentile.csv',
    'verbal_concepts' : './data/Concept Sentences for SAT Diagnostic - Verbal.csv',
    'math_concepts': './data/Concept Sentences for SAT Diagnostic - Math.csv'
}

def run_analysis(ans_dict):
    total_scale_df = pd.read_csv(data_assets['combined_score'])
    m = calculate_math_score(ans_dict)
    v = calculate_verbal_score(ans_dict)
    m.update(v)
    total_score = m.get('math_score') + v.get('verbal_score')
    m['total_score'] = total_score
    m['math_score'] = m.get('math_score')
    m['verbal_score'] = m.get('verbal_score')
    m['math_question_percent'] = m.get('math_q_percent')
    m['reading_question_percent'] = m.get('verbal_question_percent')
    m['writing_question_percent'] = m.get('writing_question_percent')
    m['total_percentile'] = total_scale_df.loc[total_scale_df['score'] == total_score ]['percentile'].tolist()[0]
    return m



def make_concept_sentences(section, concepts):
    ffile = data_assets.get('math_concepts') if section == "Math" else data_assets.get('verbal_concepts')
    df = pd.read_csv(ffile)
    df_concept = df[df.Concept.isin(concepts)]
    output = ""
    t = "".join(list(df_concept.Text))
    return t


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
    try:
        estr = eval(istr)
        if isinstance(estr, float):
            return str(round(estr, 2))
        else:
            return istr
    except:
        return ""

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

def plot_improve_barchart(dicts, key, label): # verbal_improve, math_improve
    cdicts = copy.deepcopy(dicts)
    local_dicts = cdicts.get(key)
    for x in local_dicts:
        x['improvement'] = float(x['improvement'])
    df = pd.DataFrame.from_dict(local_dicts)
    df = df.sort_values(['improvement'],ascending=False).head(6)
    objects = list(df['concept'])
    num_wrong = [float(x) for x in list(df['improvement'])]
    y_pos = np.arange(len(objects))

    plt.bar(y_pos, num_wrong, align='center', color = '#009F60')
    plt.xticks(y_pos, objects)
    plt.ylabel('Possible Score Improvement')
    plt.xlabel('Concept of Question')
    plt.title(f'SAT {label} Questions Missed By Difficulty')
    _, labels = plt.xticks(y_pos, objects, rotation='vertical')
    plt.setp(labels, rotation=45)

    ffile = make_file()
    plt.savefig(ffile, bbox_inches='tight')
    plt.close()
    return ffile

def plot_total_miss_barchart(dicts, key, label): # verbal_improve, math_improve
    cdicts = copy.deepcopy(dicts)
    local_dicts = cdicts.get(key)
    for x in local_dicts:
        x['total'] = float(x['correct']) + float(x['wrong'])
    df = pd.DataFrame.from_dict(local_dicts)
    df = df.sort_values(['wrong'],ascending=False).sort_values(['wrong'], ascending=True)
    objects     = list(df['concept'])
    num_total   = [float(x) for x in list(df['total'])]
    num_wrong   = [float(x) for x in list(df['wrong'])]
    num_correct = [float(x) for x in list(df['correct'])]
    y_pos       = np.arange(len(objects))

    p1 = plt.barh(y_pos, num_total, align='center', color = '#009F60', alpha = 0.5)
    p2 = plt.barh(y_pos, num_correct, align='center', color = '#009F60')
    plt.yticks(y_pos, objects)
    plt.xlabel('Questions Within Concept')
    plt.ylabel('Concept of Question')
    plt.title(f'SAT {label} Questions Missed')
    _, labels = plt.yticks(y_pos, objects, rotation='horizontal')
    plt.legend((p1[0], p2[0]), ('Total', 'Correct'))


    ffile = make_file()
    plt.savefig(ffile, bbox_inches='tight')
    plt.close()
    return ffile

def plot_math(dicts):
    df = pd.DataFrame.from_dict(dicts.get('math_difficulty_plot'))
    df = df.sort_values(['difficulty'])

    objects = ('Level 1 - Easy', 'Level 2 - Easy to Medium', 'Level 3 - Medium', 'Level 4 - Hard', 'Level 5 - Super-hard')
    y_pos = np.arange(len(objects))
    num_wrong = list(df['wrong'])
    num_total = list(df['total'])
    num_correct = list(df['correct'])

    p1 = plt.barh(y_pos, num_total, align='center', color = '#009F60', alpha = 0.5)
    p2 = plt.barh(y_pos, num_correct, align='center', color = '#009F60')
    plt.yticks(y_pos, objects)
    plt.xlabel('Questions Within Difficulty')
    plt.ylabel('Difficulty of Question')
    label = "Math"
    plt.title(f'SAT {label} Difficulty Missed')
    _, labels = plt.yticks(y_pos, objects, rotation='horizontal')
    plt.legend((p1[0], p2[0]), ('Total', 'Correct'))


    ffile = make_file()
    plt.savefig(ffile, bbox_inches='tight')
    plt.close()
    return ffile
    if False:
        plt.bar(y_pos, num_wrong, align='center', color = '#009F60')
        plt.xticks(y_pos, objects)
        plt.ylabel('Number of Questions')
        plt.xlabel('Difficulty Level of Question')
        plt.title('SAT Math Questions Missed By Difficulty')
        _, labels = plt.xticks(y_pos, objects, rotation='vertical')
        plt.setp(labels, rotation=45)
        ffile = make_file()
        plt.savefig(ffile, bbox_inches='tight')
        plt.close()
        return ffile


def plot_math_pie(dicts):
    df = pd.DataFrame.from_dict(dicts.get('math_difficulty'))
    df = df.sort_values(['difficulty'])
    objects = ('Level 1 - Easy', 'Level 2 - Easy to Medium', 'Level 3 - Medium', 'Level 4 - Hard', 'Level 5 - Super-hard')
    y_pos = np.arange(len(objects))
    num_wrong = list(df['wrong'])
    fig1, ax1 = plt.subplots()
    sizes = []
    labels = []
    for i in range(0,len(objects)):
        if num_wrong[i] == 0:
            next
        else:
            sizes.append(num_wrong[i])
            labels.append(objects[i])
    #sizes = num_wrong
    #labels = objects
    ax1.pie(sizes, labels=labels, shadow=False,autopct='%d', startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ffile = make_file()
    plt.title('SAT Math Questions Missed By Difficulty')
    plt.savefig(ffile, bbox_inches='tight')
    plt.close()
    return ffile


def plot_verbal(dicts):
    cols = ['difficulty', 'wrong','correct', 'total']
    v = pd.DataFrame.from_dict(dicts.get('reading_difficulty_plot'))[cols]
    w = pd.DataFrame.from_dict(dicts.get('writing_difficulty_plot'))[cols]
    ## NOTE there is no LEVEL 1 in VERBAL section
    ## so the solution here is to do an outer join
    ## then just fill the NA's with 0
    m = pd.merge(v,w, on='difficulty', how='outer')
    m.fillna(0,inplace=True)
    m['wrong'] = m.apply(lambda row: row['wrong_x'] + row['wrong_y'],axis=1)
    m['total'] = m.apply(lambda row: row['total_x'] + row['total_y'],axis=1)
    m['correct'] = m.apply(lambda row: row['correct_x'] + row['correct_y'],axis=1)
    df = m.sort_values(['difficulty'])

    objects = ('Level 1 - Easy', 'Level 2 - Easy to Medium', 'Level 3 - Medium', 'Level 4 - Hard', 'Level 5 - Super-hard')
    y_pos = np.arange(len(objects))
    num_wrong = list(df['wrong'])
    num_correct = list(df['correct'])
    num_total = list(df['total'])

    p1 = plt.barh(y_pos, num_total, align='center', color = '#009F60', alpha = 0.5)
    p2 = plt.barh(y_pos, num_correct, align='center', color = '#009F60')
    plt.yticks(y_pos, objects)
    plt.xlabel('Questions Within Difficulty')
    plt.ylabel('Difficulty of Question')
    label = "Verbal"
    plt.title(f'SAT {label} Difficulty Missed')
    _, labels = plt.yticks(y_pos, objects, rotation='horizontal')
    plt.legend((p1[0], p2[0]), ('Total', 'Correct'))


    ffile = make_file()
    plt.savefig(ffile, bbox_inches='tight')
    plt.close()
    return ffile


    if False:
        plt.bar(y_pos, num_wrong, align='center', color = '#009F60')
        _, labels = plt.xticks(y_pos, objects, rotation='vertical')
        plt.setp(labels, rotation=45)
        plt.ylabel('Number of Questions')
        plt.xlabel('Difficulty Level of Question')
        plt.title('SAT Verbal Questions Missed By Difficulty')
        ffile = make_file()
        plt.savefig(ffile, bbox_inches='tight')
        plt.close()
        return ffile

def plot_verbal_pie(dicts):
    v = pd.DataFrame.from_dict(dicts.get('reading_difficulty'))[['difficulty', 'wrong']]
    w = pd.DataFrame.from_dict(dicts.get('writing_difficulty'))[['difficulty', 'wrong']]
    m = pd.merge(v,w, on='difficulty')
    m['wrong'] = m.apply(lambda row: row['wrong_x'] + row['wrong_y'],axis=1)
    df = m.sort_values(['difficulty'])

    objects = ('Level 1 - Easy', 'Level 2 - Easy to Medium', 'Level 3 - Medium', 'Level 4 - Hard', 'Level 5 - Super-hard')
    ## NOTE there is no LEVEL 1 in VERBAL section
    y_pos = np.arange(len(objects))
    num_wrong = [0] + list(df['wrong'])

    fig1, ax1 = plt.subplots()

    sizes = []
    labels = []
    for i in range(0,len(objects)):
        if num_wrong[i] == 0:
            next
        else:
            sizes.append(num_wrong[i])
            labels.append(objects[i])

    ffile = make_file()
    if len(sizes) > 0:
        ax1.pie(sizes, labels=labels, shadow=False,autopct='%d', startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title('SAT Verbal Questions Missed By Difficulty')
        plt.savefig(ffile, bbox_inches='tight')
        plt.close()
    return ffile


def mk_concept_dict(miss, total):
    llist = [{'concept': k, \
              'correct': int(total.get(k,0)) - int(miss.get(k,0)),\
              'wrong': int(miss.get(k,0)), \
              'total': int(total.get(k,0)), \
              'pct' : fmt_percentage(miss.get(k,0),sum(total.values())) } for k in total.keys()]
    newlist = sorted(llist, key=lambda k: float(k['pct']), reverse=True)
    return newlist

def add_pct_correct_concept(concept, correct):
    output = []
    for entry in concept:
        concept = entry.get('concept')
        if concept in correct:
            entry['correct'] = correct.get(concept)
            entry['pct_correct'] = entry.get('correct') / (entry.get('wrong') + entry.get('correct'))
        else:
            entry['correct'] = 0
            entry['pct_correct'] = 0
        output = output + [entry]
    return output


def get_best_concepts(concept_list):
    perfect = [x for x in concept_list if x.get('pct_correct') > 99.999]
    if len(perfect) > 0:
        perfect_idx = len(prefect) if (len(perfect) < 3) else 3
        return [x.get('concept') for x in perfect[0:perfect_idx]]
    best = sorted(concept_list, key = lambda i: i['pct_correct'], reverse=True)[0]
    return [best.get('concept')]

def get_worst_concepts(concept_list):
    non_perfect = [x for x in concept_list if int(x.get('improvement')) > 0]
    if len(non_perfect) > 0:
        non_perfect_idx = len(non_perfect) if (len(non_perfect) < 3) else 3
        worst = sorted(non_perfect, key = lambda i: float(i['improvement']), reverse=True)[0:non_perfect_idx]
        #for x in worst:
        return [x.get('concept') for x in worst[0:non_perfect_idx]]
    else:
        return []



def mk_diff_dict(miss, total):
    return [{'difficulty': k, \
            'wrong': miss.get(k,0), \
            'correct': total.get(k,0) - miss.get(k,0),\
            'total': total.get(k,0),\
            'pct' : fmt_percentage(miss.get(k,0), sum(total.values())) }  for k in total.keys()]

def mk_diff_dict_plot(miss, total):
    return [{'difficulty': k, \
             'wrong': miss.get(k,0), \
             'total': total.get(k,0),\
             'correct':total.get(k,0) -  miss.get(k,0),
             'pct' : fmt_percentage(miss.get(k,0), sum(total.values())) }  for k in total.keys()]

def mk_explain_dict(df,section):
    df_copy = df.copy()
    df_copy['section'] = section
    return df_copy[['section', 'question', 'response', 'answer','explain']].to_dict('records')


def get_math_explain(explain_file):
    if explain_file == "":
        return ""
    if not isinstance(explain_file, str):
        return ""

    local_file = os.path.join(APP_ROOT, explain_file)
    exists = os.path.isfile(local_file)
    txt = ""
    if exists:
        with open(local_file) as f:
            txt = f.read()
    return txt


def mk_explain_dict_math(df):
    df_new = df.copy()
    df_new.loc[:,'explain2'] = df_new.apply(lambda row: get_math_explain(row['explain']), axis=1)
    df_new.loc[:,'explain'] = df_new['explain2']
    df_new.loc[:,'section'] = df_new.apply(lambda row: "Math Calculator" if row['section'] == "Math1" else "Math No Calculator",axis=1)
    ddict = df_new[['section', 'question', 'response', 'answer','explain']].to_dict('records')
    return(ddict)


def mk_improve_dict(miss, total):
    total_N = float(sum(total.values()))
    llist = [{'concept': k, 'improvement': fmt_improve((float(miss.get(k,0))/float(total.get(k,0))) * (float(total.get(k,0))/total_N) * 800) } for k in total.keys() ]
    newlist = sorted(llist, key = lambda k: float(k['improvement']), reverse=True)
    return newlist



def calc_best_worst_difficulty(diff_list, diff_total):
    improve = [{'improve' : x.get('wrong') * float(x.get('pct')),  \
        'difficulty' : x.get('difficulty'), \
        'pct_correct' : 1 - (x.get('wrong') / diff_total.get(x.get('difficulty')))}\
        for x in diff_list]
    improve_list = sorted(improve, key = lambda k: float(k['improve']), reverse=True)
    most_improve_level = improve_list[0].get('difficulty')

    strongest_list = sorted(improve, key = lambda k: float(k['pct_correct']), reverse=True)
    most_improve_level = improve_list[0].get('difficulty')
    strongest_level = strongest_list[0].get('difficulty')
    return {'improve':most_improve_level, 'strong':strongest_level}


def filter_concepts_top5(cdict):
    idx = 5 if (len(cdict) > 5) else len(cdict)
    return(cdict[0:idx])

def pop_pct(ddict):
    for x in ddict:
        if x.get('pct'):
            x.pop('pct')


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
    math_q_percent = int(100 * total_correct / (38+20) )

    score = int(m_score_df.loc[m_score_df['correct_ans'] == total_correct]['score'].tolist()[0])
    percentile = int(m_score_df.loc[m_score_df['correct_ans'] == total_correct]['percentile'].tolist()[0])

    # Match combined df
    m1_ans_df.loc[:,"section"] = "Math1"
    m2_ans_df.loc[:,"section"] = "Math2"
    m_ans_df = pd.concat([m1_ans_df.copy(), m2_ans_df.copy()])

    # Math Combined Incorrect Explainations
    m_explain_dict = mk_explain_dict_math(m_ans_df.loc[m_ans_df['correct'] == False])

    # Math concepts
    m_total_concepts = agg_counts_dict(m_ans_df[['concept','concept2']])
    m_missed_concepts = agg_counts_dict(m_ans_df.loc[m_ans_df['correct'] == False][['concept','concept2']])
    m_correct_concepts = agg_counts_dict(m_ans_df.loc[m_ans_df['correct'] == True][['concept','concept2']])
    #m_concept_dict =  mk_concept_dict(m_missed_concepts, m_total_concepts)
    m_concept_dict_table = mk_concept_dict(m_missed_concepts, m_total_concepts)
    pop_pct(m_concept_dict_table)
    m_concept_dict =  add_pct_correct_concept(mk_concept_dict(m_missed_concepts, m_total_concepts),
                                              m_correct_concepts)
    m_improve_dict = mk_improve_dict(m_missed_concepts, m_total_concepts)

    # Math Difficulty
    m_missed_diff = dict(m_ans_df.loc[m_ans_df['correct'] == False][['difficulty']].apply(pd.value_counts)['difficulty'])
    m_total_diff = agg_counts_dict(m_ans_df[['difficulty']])
    m_diff_dict = mk_diff_dict(m_missed_diff, m_total_diff)
    pop_pct(m_diff_dict)
    m_diff_dict_plot = mk_diff_dict_plot(m_missed_diff, m_total_diff)


    math_best_concepts = get_best_concepts(m_concept_dict)
    math_worst_concepts = get_worst_concepts(m_improve_dict)
    math_improve_stmt = make_concept_sentences('Math', math_worst_concepts)

    odict = {
        'math_score': score,
        'math_percentile': percentile,
        'math_concepts': m_concept_dict_table,
        'math_difficulty': m_diff_dict,
        'math_difficulty_plot': m_diff_dict_plot,
        'math_explain' : m_explain_dict,
        'math_improve' : filter_concepts_top5(m_improve_dict),
        'math_q_percent': math_q_percent,
        'math_best_concepts' : math_best_concepts,
        'math_worst_concepts' : math_worst_concepts,
        'math_improve_stmt': math_improve_stmt,
        'math_concept_plot' : m_concept_dict
    }
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
    v_explain_dict = mk_explain_dict(v_ans_df.loc[v_ans_df['correct'] == False], 'Reading')

    v_score_df = pd.read_csv(data_assets.get('verbal_score'))
    verbal_raw_score = v_score_df.loc[v_score_df['correct_ans'] == verbal_num_correct]['reading_raw_score'].tolist()[0]
    writing_raw_score = int(v_score_df.loc[v_score_df['correct_ans'] == writing_num_correct]['writing_raw_score'].tolist()[0])

    v_scale_df = pd.read_csv(data_assets.get('verbal_scale'))
    verbal_question_percent = int(100 * verbal_num_correct / 52)
    writing_question_percent = int(100 * writing_num_correct / 44)
    raw_score = verbal_raw_score + writing_raw_score

    score = int(v_scale_df.loc[v_scale_df['raw'] == raw_score ]['score'].tolist()[0])
    percentile = int(v_scale_df.loc[v_scale_df['raw'] == raw_score ]['percentile'].tolist()[0])

    # Verbal Concepts
    v_total_concepts = agg_counts_dict(v_ans_df[['concept','concept2','concept3']])
    v_missed_concepts = agg_counts_dict(v_ans_df.loc[v_ans_df['correct'] == False][['concept','concept2','concept3']])
    v_correct_concepts = agg_counts_dict(v_ans_df.loc[v_ans_df['correct'] == True][['concept','concept2','concept3']])

    v_concept_dict_table = mk_concept_dict(v_missed_concepts, v_total_concepts)
    pop_pct(v_concept_dict_table)
    v_concept_dict =  add_pct_correct_concept(mk_concept_dict(v_missed_concepts, v_total_concepts),
                                              v_correct_concepts)
    v_sum = sum([float(x.get('pct')) for x in v_concept_dict])
    # Writing Concepts

    w_total_concepts = agg_counts_dict(w_ans_df[['concept','concept2']])
    w_missed_concepts = agg_counts_dict(w_ans_df.loc[w_ans_df['correct'] == False][['concept','concept2']])
    w_correct_concepts = agg_counts_dict(w_ans_df.loc[w_ans_df['correct'] == True][['concept','concept2']])
    w_concept_dict_table = mk_concept_dict(w_missed_concepts, w_total_concepts)
    pop_pct(w_concept_dict_table)
    w_concept_dict =  add_pct_correct_concept(mk_concept_dict(w_missed_concepts, w_total_concepts),
                                              w_correct_concepts)
    w_sum = sum([float(x.get('pct')) for x in w_concept_dict])

    vw_best_concepts = get_best_concepts(w_concept_dict + v_concept_dict)
    vw_concept_dict = w_concept_dict + v_concept_dict

    # Verbal Difficulty
    v_missed_diff = dict(v_ans_df.loc[v_ans_df['correct'] == False][['difficulty']].apply(pd.value_counts)['difficulty'])
    v_total_diff = agg_counts_dict(v_ans_df[['difficulty']])
    v_diff_dict = mk_diff_dict(v_missed_diff, v_total_diff)
    v_diff_dict_plot = mk_diff_dict_plot(v_missed_diff, v_total_diff)

    # Writing Difficulty
    w_missed_diff = dict(w_ans_df.loc[w_ans_df['correct'] == False][['difficulty']].apply(pd. value_counts)['difficulty'])
    w_total_diff = agg_counts_dict(w_ans_df[['difficulty']])
    w_diff_dict = mk_diff_dict(w_missed_diff, w_total_diff)
    w_diff_dict_plot = mk_diff_dict_plot(w_missed_diff, w_total_diff)


    # Verbal Section Improvement
    vw_total_concepts = merge_two_dicts(v_total_concepts, w_total_concepts)
    vw_missed_concepts = merge_two_dicts(v_missed_concepts, w_missed_concepts)
    vw_improve_dict = mk_improve_dict(vw_missed_concepts, vw_total_concepts)

    vw_worst_concepts = get_worst_concepts(vw_improve_dict)

    verbal_improve_stmt = make_concept_sentences('Verbal',vw_worst_concepts)
    v_improve_dlevel = calc_best_worst_difficulty(v_diff_dict, v_total_diff)

    pop_pct(v_diff_dict)
    pop_pct(w_diff_dict)
    odict = {
        'verbal_score': score,
        'verbal_percentile': percentile,
        'reading_concepts': v_concept_dict_table,
        'reading_difficulty': v_diff_dict,
        'reading_difficulty_plot': v_diff_dict_plot,
        'reading_explain' : v_explain_dict,
        'writing_concepts': w_concept_dict_table,
        'writing_difficulty': w_diff_dict,
        'writing_difficulty_plot': w_diff_dict_plot,
        'writing_explain' : w_explain_dict,
        'verbal_improve' : filter_concepts_top5(vw_improve_dict),
        'verbal_question_percent' : verbal_question_percent,
        'writing_question_percent' : writing_question_percent,
        'verbal_best_concepts': vw_best_concepts,
        'verbal_worst_concepts' : vw_worst_concepts,
        'verbal_improve_stmt': verbal_improve_stmt,
        'verbal_concept_plot': vw_concept_dict,
        'reading_concept_plot': v_concept_dict,
        'writing_concept_plot': w_concept_dict
    }

    return(odict)


def format_profile_table(diff_dicts):
    table = ItemTable(diff_dicts)
    return(table)
