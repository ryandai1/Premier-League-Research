"""Rebuild the audited EPL training release from frozen raw snapshots.
Run from any directory: python scripts/build_dataset.py
Only rows dated strictly before each fixture may contribute to its features.
"""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ALIASES = {'Sheffield':'Sheffield United', "Nott'm Forest":'Nottingham Forest', 'Nottingham':'Nottingham Forest', 'Man City':'Manchester City', 'Man United':'Manchester United'}
MAP_R = {'HomeGoal':'home_goals','AwayGoal':'away_goals','HomeShot':'home_shots','AwayShot':'away_shots','HomeOnTargetShot':'home_sot','AwayOnTargetShot':'away_sot','HomeCorner':'home_corners','AwayCorner':'away_corners','HomeFoul':'home_fouls','AwayFoul':'away_fouls','HomeYellow':'home_yellow','AwayYellow':'away_yellow','HomeRed':'home_red','AwayRed':'away_red','XGHome':'home_xg','XGAway':'away_xg'}
MAP_T = {'HomeGoals':'home_goals','AwayGoals':'away_goals','HomeShots':'home_shots','AwayShots':'away_shots','HomeShotsOnTarget':'home_sot','AwayShotsOnTarget':'away_sot','HomeCorners':'home_corners','AwayCorners':'away_corners','HomeFouls':'home_fouls','AwayFouls':'away_fouls','HomeYellowCards':'home_yellow','AwayYellowCards':'away_yellow','HomeRedCards':'home_red','AwayRedCards':'away_red'}
MAP_F = {'FTHG':'home_goals','FTAG':'away_goals','HS':'home_shots','AS':'away_shots','HST':'home_sot','AST':'away_sot','HC':'home_corners','AC':'away_corners','HF':'home_fouls','AF':'away_fouls','HY':'home_yellow','AY':'away_yellow','HR':'home_red','AR':'away_red'}
KEY = ['season_start','home_team','away_team']
METRICS = ['goals_for','goals_against','shots_for','shots_against','sot_for','sot_against','corners_for','points']
FEATURES = [f'{side}_{metric}_mean{w}' for side in ['home','away'] for w in [5,10] for metric in METRICS] + [f'{side}_{f}' for side in ['home','away'] for f in ['prior_matches_365d_capped10','days_since_previous_epl','elo']] + ['elo_home_minus_away']
XG_FEATURES = [f'{side}_xg_{direction}_mean{w}' for side in ['home','away'] for direction in ['for','against'] for w in [5,10]]


def canonical(raw, source, season=None):
    d = raw.rename(columns={'HomeTeam':'home_team','AwayTeam':'away_team'}).copy()
    for c in ['home_team','away_team']:
        d[c] = d[c].str.strip().replace(ALIASES)
    if source == 'reza':
        d['date'] = pd.to_datetime(d[['Year','Month','Day']].rename(columns=str.lower),errors='raise')
        d['season_start'] = 2000 + d.Season.astype(str).str.zfill(4).str[:2].astype(int)
        d = d.rename(columns=MAP_R)
        result = d.Result
        metrics = list(MAP_R.values())
    elif source == 'tara':
        d['date'] = pd.to_datetime(d.Date,format='%Y-%m-%d',errors='raise')
        d['season_start'] = d.Season.str[:4].astype(int)
        d = d.rename(columns=MAP_T)
        result = d.FullTime.map({'HomeWin':'H','AwayWin':'A','Draw':'D'})
        metrics = list(MAP_T.values())
    else:
        d['date'] = pd.to_datetime(d.Date,dayfirst=True,errors='raise')
        d['season_start'] = int(season)
        d = d.rename(columns=MAP_F)
        result = d.FTR
        metrics = list(MAP_F.values())
    for c in metrics:
        d[c] = pd.to_numeric(d[c],errors='raise')
    d['result'] = np.select([d.home_goals>d.away_goals,d.home_goals<d.away_goals],['H','A'],default='D')
    assert (d.result == result).all(), f'{source}: labels disagree with goals'
    assert not d.duplicated(KEY).any(), f'{source}: duplicate fixture key'
    assert (d.home_team!=d.away_team).all()
    assert (d.date.dt.year.ge(d.season_start)&d.date.dt.year.le(d.season_start+1)).all()
    for c in metrics:
        assert d[c].dropna().ge(0).all(), c
        if not c.endswith('xg'):
            assert d[c].dropna().mod(1).eq(0).all(), c
    return d[KEY+['date','result']+metrics]


def make_features(matches):
    """Day-batched updates avoid relying on historical kickoff-time accuracy.
    Histories use last 5/10 EPL fixtures within 365 days. A metric needs >=5
    nonmissing values. Elo: start 1500, K=20, home advantage=60; annual
    shrink 0.75; reset to 1500 after >365 days absent. Fixed, untuned constants.
    """
    history = defaultdict(list)
    elo = defaultdict(lambda:1500.)
    last = {}
    out = []
    last_season = None
    for date,day in matches.sort_values(['date','match_id']).groupby('date',sort=True):
        season = int(day.season_start.iloc[0])
        assert day.season_start.nunique()==1
        if last_season is not None and season!=last_season:
            elo = defaultdict(lambda:1500., {team:1500.+.75*(value-1500.) for team,value in elo.items()})
        last_season = season
        teams = list(day.home_team)+list(day.away_team)
        assert len(teams)==len(set(teams)), 'A club plays twice on one day'
        for team in teams:
            if team in last and (date-last[team]).days>365:
                elo[team]=1500.
        pending = []
        for m in day.to_dict('records'):
            row = {'match_id':m['match_id'],'date':date,'season_start':season,'home_team':m['home_team'],'away_team':m['away_team']}
            for side,other in [('home','away'),('away','home')]:
                team=m[f'{side}_team']
                h=[v for v in history[team] if 0<(date-v['date']).days<=365][-10:]
                row[f'{side}_prior_matches_365d_capped10']=len(h)
                row[f'{side}_history_latest_date']=h[-1]['date'] if h else pd.NaT
                row[f'{side}_days_since_previous_epl']=(date-last[team]).days if team in last else np.nan
                row[f'{side}_elo']=elo[team]
                for w in [5,10]:
                    for metric in METRICS+['xg_for','xg_against']:
                        values=[v[metric] for v in h[-w:] if pd.notna(v[metric])]
                        row[f'{side}_{metric}_mean{w}']=float(np.mean(values)) if len(values)>=5 else np.nan
                gf,ga=m[f'{side}_goals'],m[f'{other}_goals']
                event={'date':date,'goals_for':gf,'goals_against':ga,'shots_for':m[f'{side}_shots'],'shots_against':m[f'{other}_shots'],'sot_for':m[f'{side}_sot'],'sot_against':m[f'{other}_sot'],'corners_for':m[f'{side}_corners'],'points':3 if gf>ga else 1 if gf==ga else 0,'xg_for':m[f'{side}_xg'],'xg_against':m[f'{other}_xg']}
                pending.append((team,event))
            row['elo_home_minus_away']=row['home_elo']-row['away_elo']
            row['target_result']=m['result']
            row['target_home_goals']=m['home_goals']
            row['target_away_goals']=m['away_goals']
            out.append(row)
        # Update ONLY after every prediction row for this date has been generated.
        for m in day.to_dict('records'):
            h,a=m['home_team'],m['away_team']
            p=1/(1+10**((elo[a]-elo[h]-60)/400))
            observed=1 if m['home_goals']>m['away_goals'] else 0 if m['home_goals']<m['away_goals'] else .5
            delta=20*(observed-p);elo[h]+=delta;elo[a]-=delta
        for team,event in pending:
            history[team].append(event);history[team]=history[team][-10:];last[team]=date
    return pd.DataFrame(out)


def main():
    for folder in ['data','audit']:(ROOT/folder).mkdir(exist_ok=True)
    manifest=json.loads((ROOT/'source_manifest.json').read_text())
    for entry in manifest:
        p=ROOT/'raw'/(entry['name']+'.csv')
        assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256'],f'Snapshot changed: {p}'
    raw=pd.read_csv(ROOT/'raw/reza.csv',dtype={'Season':str})
    traw=pd.read_csv(ROOT/'raw/tara.csv')
    r=canonical(raw,'reza');t=canonical(traw,'tara')
    overlap=r.merge(t,on=KEY,suffixes=('_r','_t'),validate='one_to_one',how='inner')
    assert len(overlap)==len(t)==3800
    assert overlap.date_r.eq(overlap.date_t).all()
    assert overlap.result_r.eq(overlap.result_t).all()
    conflicts=[]
    for c in MAP_T.values():
        bad=overlap[f'{c}_r'].notna()&overlap[f'{c}_t'].notna()&overlap[f'{c}_r'].ne(overlap[f'{c}_t'])
        for _,v in overlap[bad].iterrows():
            conflicts.append({**{k:v[k] for k in KEY},'field':c,'reza_value':v[f'{c}_r'],'tara_value':v[f'{c}_t']})
    cross=[]
    for code in ['1516','2425','2526']:
        fd=canonical(pd.read_csv(ROOT/'raw'/f'football_data_{code}.csv'),'football_data',2000+int(code[:2]))
        m=r[r.season_start.eq(2000+int(code[:2]))].merge(fd,on=KEY,suffixes=('_r','_f'),validate='one_to_one')
        assert len(m)==380
        for c in ['date','result']+list(MAP_F.values()):
            assert m[f'{c}_r'].eq(m[f'{c}_f']).all(),f'football-data conflict: {code} {c}'
        cross.append({'season':code,'matched':len(m),'fields_compared':['date','result']+list(MAP_F.values()),'disagreements':0})
    assert len(conflicts)==1 and conflicts[0]['field']=='home_shots'
    conflicts[0].update(resolution='Keep Reza value 15: football-data 2015/16 confirms 15; Tara gives 5, below 8 shots on target',chosen_source='reza_verified_football_data')
    pd.DataFrame(conflicts).to_csv(ROOT/'audit/source_conflicts.csv',index=False)
    (ROOT/'audit/football_data_crosscheck.json').write_text(json.dumps(cross,indent=2))
    r=r.sort_values(['date','home_team','away_team']).reset_index(drop=True)
    r['match_id']=[hashlib.sha256(f'{int(v.season_start)}|{v.home_team}|{v.away_team}'.encode()).hexdigest()[:16] for v in r.itertuples()]
    assert r.match_id.is_unique
    r['season']=r.season_start.map(lambda x:f'{x}/{str(x+1)[-2:]}')
    r['shared_with_tara']=r.season_start.between(2010,2019)
    r['source']='reza'
    r['date_and_core_stats_crosschecked_football_data']=r.season_start.isin([2015,2024,2025])
    issues=[]
    for side in ['home','away']:
        bad=r[f'{side}_sot']>r[f'{side}_shots']
        r[f'{side}_shots_quality_flag']=bad
        for _,v in r[bad].iterrows():
            issues.append({'match_id':v.match_id,'date':str(v.date.date()),'home_team':v.home_team,'away_team':v.away_team,'side':side,'shots':v[f'{side}_shots'],'shots_on_target':v[f'{side}_sot'],'action':'Set both contradictory measurements to missing; preserve match and goals'})
        r.loc[bad,[f'{side}_shots',f'{side}_sot']]=np.nan
    pd.DataFrame(issues).to_csv(ROOT/'audit/invalid_shot_measurements.csv',index=False)
    r.to_csv(ROOT/'data/matches_clean.csv',index=False)
    counts=[]
    for season,g in r.groupby('season_start'):
        n=pd.concat([g.home_team,g.away_team]).value_counts()
        assert len(g)==380 and len(n)==20 and n.eq(38).all()
        counts.append({'season_start':season,'matches':len(g),'teams':len(n),'arsenal_matches':int((g.home_team.eq('Arsenal')|g.away_team.eq('Arsenal')).sum()),'matches_with_both_xg':int(g[['home_xg','away_xg']].notna().all(axis=1).sum())})
    pd.DataFrame(counts).to_csv(ROOT/'audit/season_coverage.csv',index=False)
    f=make_features(r)
    f['split']=np.select([f.season_start<=2021,f.season_start<=2023],['train','validation'],default='test')
    f['core_ready']=f[FEATURES].notna().all(axis=1)
    f['xg_ready']=f[XG_FEATURES].notna().all(axis=1)
    f['exclusion_reason']=np.where(f.core_ready,'','Insufficient prior EPL history or fewer than 5 valid measurements in a required window')
    for side in ['home','away']:
        date=f[f'{side}_history_latest_date']
        assert (date.isna()|date.lt(f.date)).all()
    assert np.isfinite(f.loc[f.core_ready,FEATURES].to_numpy(dtype=float)).all()
    f.to_csv(ROOT/'data/prematch_all.csv',index=False)
    f.loc[~f.core_ready,['match_id','date','home_team','away_team','split','exclusion_reason']].to_csv(ROOT/'audit/excluded_from_core.csv',index=False)
    metadata=['match_id','date','season_start','home_team','away_team','split']
    targets=['target_result','target_home_goals','target_away_goals']
    ready=f[f.core_ready].copy()
    for split,g in ready.groupby('split'):
        folder=ROOT/'data'/split;folder.mkdir(exist_ok=True)
        g[FEATURES].to_csv(folder/'X.csv',index=False)
        g[targets].to_csv(folder/'y.csv',index=False)
        g[metadata].to_csv(folder/'index.csv',index=False)
    arsenal=ready[ready.home_team.eq('Arsenal')|ready.away_team.eq('Arsenal')].copy()
    arsenal['arsenal_is_home']=arsenal.home_team.eq('Arsenal')
    arsenal['arsenal_result']=np.where(arsenal.target_result.eq('D'),'D',np.where(arsenal.target_result.eq(np.where(arsenal.arsenal_is_home,'H','A')),'W','L'))
    arsenal.to_csv(ROOT/'data/arsenal_view.csv',index=False)
    schema={'unit':'one unique EPL match; home/away orientation','features':FEATURES,'optional_xg_features':XG_FEATURES,'targets':targets,'class_labels':{'H':'home win','D':'draw','A':'away win'},'metadata':metadata,'prediction_cutoff':'start of match date; only strictly earlier dates enter histories','core_ready':'all 39 default numeric features observed; both sides require at least 5 recent EPL matches and >=5 valid values per metric window','xg_note':'Optional only; never zero-impute unavailable historical xG. Provider continuity not established.','time_split':{'train':'2000/01–2021/22','validation':'2022/23–2023/24','test':'2024/25–2025/26'},'numeric_missing_policy':'X split files have no missing values; excluded fixtures retained in prematch_all.csv','model_input_warning':'Select only features in this allowlist. Neither metadata nor current-match targets/observations are predictors.'}
    (ROOT/'feature_schema.json').write_text(json.dumps(schema,indent=2))
    lineage=[]
    for c in raw.columns:
        if c in ['Season','Year','Month','Day','HomeTeam','AwayTeam']:
            action='Metadata / chronological indexing; not direct model feature'
        elif c in MAP_R:
            action='Post-match observation; retained for targets and strictly lagged histories only'
        elif c=='Result':
            action='Target; independently recomputed from goals'
        else:
            action='Excluded from default model: post-match/final-season information or unverified historical availability/definition'
        lineage.append({'raw_column':c,'handling':action})
    pd.DataFrame(lineage).to_csv(ROOT/'audit/raw_column_handling.csv',index=False)
    summary={'raw_reza_rows':len(raw),'raw_reza_columns':len(raw.columns),'readme_claimed_columns':65,'raw_tara_rows':len(traw),'matched_overlap':len(overlap),'additional_matches_from_tara':0,'unique_matches':len(r),'seasons':int(r.season_start.nunique()),'teams':int(pd.concat([r.home_team,r.away_team]).nunique()),'arsenal_total_matches':int((r.home_team.eq('Arsenal')|r.away_team.eq('Arsenal')).sum()),'source_measurement_conflicts':len(conflicts),'contradictory_shot_matches':len(set(v['match_id'] for v in issues)),'contradictory_shot_team_sides':len(issues),'matches_missing_either_xg':int(r[['home_xg','away_xg']].isna().any(axis=1).sum()),'core_features':len(FEATURES),'core_ready_matches':len(ready),'core_excluded_matches':int((~f.core_ready).sum()),'split_rows':{str(k):int(v) for k,v in ready.groupby('split').size().items()},'arsenal_ready_matches':len(arsenal),'arsenal_split_rows':{str(k):int(v) for k,v in arsenal.groupby('split').size().items()},'optional_xg_ready_matches':int((f.core_ready&f.xg_ready).sum()),'min_date':str(r.date.min().date()),'max_date':str(r.date.max().date()),'test_model_evaluation_performed':False}
    (ROOT/'audit/summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
