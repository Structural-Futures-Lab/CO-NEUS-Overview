"""
Colorado Building Data Processor - Overview Only Version
Processes the NSI-Enhanced USA Structures Dataset for Colorado.
Includes: overview stats, temporal, hierarchical distribution (no drainage/soil),
Year→Occ→Material→Foundation flow, occupancy hierarchy, OCC_CLS→occtype sankey, MIX_SC.
"""
import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings
import re
from collections import Counter

warnings.filterwarnings('ignore')


class COBuildingDataProcessor:
    def __init__(self, file_path='usa_structures_nsi_with_height.gpkg'):
        self.file_path = file_path
        self.df = None
        self.df_cleaned = None
        self.data_flow_stats = {}

    # ───────────────────── LOAD ─────────────────────
    def load_data(self):
        import pyogrio
        print(f"Loading data from {self.file_path} …")
        try:
            layers = pyogrio.list_layers(self.file_path)
            print(f"Available layers: {layers}")
            import geopandas as gpd
            layer_name = layers[0][0] if len(layers) > 0 else None
            print(f"Loading layer: {layer_name}")
            self.gdf = gpd.read_file(self.file_path, layer=layer_name)
            print(f"Loaded {len(self.gdf)} records.  CRS={self.gdf.crs}")
            print(f"Columns: {list(self.gdf.columns)}")
            if 'LATITUDE' not in self.gdf.columns or 'LONGITUDE' not in self.gdf.columns:
                print("Extracting LATITUDE/LONGITUDE from geometry …")
                gdf4326 = self.gdf.to_crs(epsg=4326) if (self.gdf.crs and self.gdf.crs.to_epsg() != 4326) else self.gdf
                self.gdf['LONGITUDE'] = gdf4326.geometry.centroid.x
                self.gdf['LATITUDE']  = gdf4326.geometry.centroid.y
            self.df = pd.DataFrame(self.gdf.drop(columns='geometry', errors='ignore'))
        except Exception as e:
            print(f"geopandas failed ({e}), trying pyogrio without geometry …")
            self.df = pyogrio.read_dataframe(self.file_path, ignore_geometry=True, use_arrow=True)
            print(f"Loaded {len(self.df)} records (no geometry).")
        self.data_flow_stats['initial_count'] = len(self.df)
        self.data_flow_stats['initial_columns'] = list(self.df.columns)
        self._normalize_columns()
        return self

    def _normalize_columns(self):
        existing = set(self.df.columns)
        alternatives = {
            'OCC_CLS':          ['occ_cls','OCCCLASS','occ_class'],
            'year_built':       ['YEAR_BUILT','YR_BUILT','yr_built','YRBUILT'],
            'Est GFA sqmeters': ['EST_GFA_SQMETERS','est_gfa_sqmeters','GFA_SQMETERS'],
            'HEIGHT':           ['height','Height','BLDG_HT'],
            'PRED_HEIGHT':      ['pred_height','Pred_Height','PREDICTED_HEIGHT'],
            'SQMETERS':         ['sqmeters','SqMeters','FOOTPRINT_SQMETERS'],
            'material_type':    ['MATERIAL_TYPE','MAT_TYPE','mat_type','BLDGTYPE'],
            'foundation_type':  ['FOUNDATION_TYPE','FOUND_TYPE','found_type'],
            'LATITUDE':         ['latitude','Latitude','LAT','lat'],
            'LONGITUDE':        ['longitude','Longitude','LON','lon'],
            'PRIM_OCC':         ['prim_occ','PrimOcc','PRIMOCC'],
            'OCC_DICT':         ['occ_dict','OCCDICT'],
            'MIX_SC':           ['mix_sc','MIXSC'],
        }
        col_map = {}
        for target, candidates in alternatives.items():
            if target not in existing:
                for alt in candidates:
                    if alt in existing:
                        col_map[alt] = target; break
        if col_map:
            print(f"Renaming columns: {col_map}")
            self.df.rename(columns=col_map, inplace=True)
        for c in ['OCC_CLS','year_built','LATITUDE','LONGITUDE']:
            if c not in self.df.columns: print(f"  WARNING: Required column '{c}' not found!")
        for c in ['Est GFA sqmeters','HEIGHT','PRED_HEIGHT','SQMETERS','material_type','foundation_type','PRIM_OCC','OCC_DICT','MIX_SC']:
            if c not in self.df.columns: print(f"  INFO: Optional column '{c}' not found.")

    # ───────────────────── CLEAN ─────────────────────
    def clean_data(self):
        print("Cleaning data …")
        cs = {'initial_count': len(self.df)}
        self.df_cleaned = self.df.copy()

        # Year
        if 'year_built' in self.df_cleaned.columns:
            m = self.df_cleaned['year_built'] > 0
            cs['invalid_year_count'] = int((~m).sum())
            self.df_cleaned = self.df_cleaned[m].copy()
        else:
            cs['invalid_year_count'] = 0
        cs['after_year_filter'] = len(self.df_cleaned)

        # Material/Foundation
        n = 0
        if 'material_type' in self.df_cleaned.columns and 'foundation_type' in self.df_cleaned.columns:
            mm = self.df_cleaned['material_type'].isna() | self.df_cleaned['foundation_type'].isna()
            n = int(mm.sum())
            if n: self.df_cleaned = self.df_cleaned[~mm].copy()
        cs['missing_mat_found_count'] = n
        cs['after_mat_found_filter'] = len(self.df_cleaned)

        # Area
        n = 0
        if 'Est GFA sqmeters' in self.df_cleaned.columns:
            mm = self.df_cleaned['Est GFA sqmeters'].isna() | (self.df_cleaned['Est GFA sqmeters'] <= 0)
            n = int(mm.sum())
            if n: self.df_cleaned = self.df_cleaned[~mm].copy()
        elif 'SQMETERS' in self.df_cleaned.columns:
            print("  Using SQMETERS as fallback for Est GFA sqmeters")
            self.df_cleaned['Est GFA sqmeters'] = self.df_cleaned['SQMETERS']
            mm = self.df_cleaned['Est GFA sqmeters'].isna() | (self.df_cleaned['Est GFA sqmeters'] <= 0)
            n = int(mm.sum())
            if n: self.df_cleaned = self.df_cleaned[~mm].copy()
        cs['missing_area_count'] = n
        cs['after_missing_area'] = len(self.df_cleaned)

        # Height
        ihc = 0
        if 'HEIGHT' in self.df_cleaned.columns:
            h = pd.to_numeric(self.df_cleaned['HEIGHT'], errors='coerce')
            bad = h.notna() & (h <= 0)
            ihc = int(bad.sum())
            if ihc: self.df_cleaned = self.df_cleaned[~bad].copy()
        h  = pd.to_numeric(self.df_cleaned['HEIGHT'], errors='coerce') if 'HEIGHT' in self.df_cleaned.columns else pd.Series(np.nan, index=self.df_cleaned.index)
        ph = pd.to_numeric(self.df_cleaned['PRED_HEIGHT'], errors='coerce') if 'PRED_HEIGHT' in self.df_cleaned.columns else pd.Series(np.nan, index=self.df_cleaned.index)
        self.df_cleaned['HEIGHT_USED'] = np.where(h.notna() & (h > 0), h, ph)
        bad2 = self.df_cleaned['HEIGHT_USED'].isna() | (self.df_cleaned['HEIGHT_USED'] <= 0)
        iahc = int(bad2.sum())
        if iahc: self.df_cleaned = self.df_cleaned[~bad2].copy()

        cs['invalid_raw_height_count'] = ihc
        cs['invalid_assumed_height_count'] = iahc
        cs['after_height_filter'] = len(self.df_cleaned)
        cs['final_count'] = len(self.df_cleaned)
        cs['total_removed'] = cs['initial_count'] - cs['final_count']
        cs['removal_percentage'] = round(cs['total_removed'] / cs['initial_count'] * 100, 2) if cs['initial_count'] else 0
        self.data_flow_stats['cleaning_pipeline'] = cs
        self.data_flow_stats['cleaning_stats'] = cs
        print(f"  Cleaned: {len(self.df_cleaned):,} records  (removed {cs['total_removed']:,} = {cs['removal_percentage']}%)")
        return self

    # ───────────────────── OVERVIEW COUNTS ─────────────────────
    def get_overview_occupancy_counts(self):
        print("Calculating overview occupancy counts …")
        if 'OCC_CLS' not in self.df_cleaned.columns: return {}
        return self.df_cleaned['OCC_CLS'].value_counts().to_dict()

    # ───────────────────── TEMPORAL ─────────────────────
    def process_temporal_data(self):
        print("Processing temporal data …")
        if 'year_built' not in self.df_cleaned.columns or 'OCC_CLS' not in self.df_cleaned.columns:
            return []
        td = []
        dv = self.df_cleaned[(self.df_cleaned['year_built'] > 1600) & (self.df_cleaned['year_built'] <= 2030)]
        for year in sorted(dv['year_built'].unique()):
            yd = dv[dv['year_built'] == year]
            for oc in yd['OCC_CLS'].unique():
                od = yd[yd['OCC_CLS'] == oc]
                aa = float(od['Est GFA sqmeters'].mean()) if 'Est GFA sqmeters' in od.columns and not od['Est GFA sqmeters'].isna().all() else 0
                ta = float(od['Est GFA sqmeters'].sum()) if 'Est GFA sqmeters' in od.columns else 0
                td.append({'year': int(year), 'display_year': str(int(year)), 'occupancy': oc, 'count': len(od), 'avg_area': aa, 'total_area': ta})
        return td

    # ───────────────────── HIERARCHICAL DISTRIBUTION (no drainage) ─────────────────────
    def _process_hierarchy(self, df, levels, value_mode='count', root_name='All Buildings'):
        sankey_data = {'nodes': [], 'links': []}
        df_proc = df.copy()
        if 'occ_cat' in levels:
            oc = df_proc['OCC_CLS'].value_counts()
            top9 = oc.nlargest(9).index.tolist()
            df_proc['occ_cat'] = df_proc['OCC_CLS'].apply(lambda x: x if x in top9 else 'Other')
        full_hierarchy = [root_name] + levels
        for i in range(len(full_hierarchy) - 1):
            src_lvl = full_hierarchy[i]; tgt_lvl = full_hierarchy[i + 1]
            gcols = [src_lvl, tgt_lvl] if i > 0 else [tgt_lvl]
            if value_mode == 'gfa':
                ag = df_proc.groupby(gcols, observed=True).agg({'Est GFA sqmeters': 'sum'}).reset_index()
                ag.rename(columns={'Est GFA sqmeters': 'value'}, inplace=True)
            else:
                ag = df_proc.groupby(gcols, observed=True).size().reset_index(name='value')
            for _, row in ag.iterrows():
                sankey_data['links'].append({'source': str(row.get(src_lvl, root_name)), 'target': str(row[tgt_lvl]), 'value': row['value']})
        node_names = set()
        for l in sankey_data['links']:
            node_names.add(l['source']); node_names.add(l['target'])
        node_map = {n: {'name': n} for n in node_names}
        for i, lname in enumerate(full_hierarchy):
            if lname in df_proc.columns:
                for cat in df_proc[lname].unique():
                    if str(cat) in node_map: node_map[str(cat)]['level'] = i
        if root_name in node_map: node_map[root_name]['level'] = 0
        sankey_data['nodes'] = list(node_map.values())
        sankey_data['total_buildings'] = len(df)
        return sankey_data

    def process_hierarchical_distribution(self):
        """Occupancy → Area → Height → Year (no drainage)."""
        print("Processing hierarchical distribution (no drainage) …")
        df_work = self.df_cleaned.copy()
        if 'Est GFA sqmeters' not in df_work.columns or 'HEIGHT_USED' not in df_work.columns:
            print("  Skipping: missing area or height columns."); return {}
        ap = df_work['Est GFA sqmeters'].quantile([0.33, 0.67]).values
        area_bins = [0, ap[0], ap[1], float('inf')]
        area_labels = ['Small', 'Medium', 'Large']
        hp = df_work['HEIGHT_USED'].dropna().quantile([0.33, 0.67]).values
        height_bins = [0, hp[0], hp[1], float('inf')]
        height_labels = ['Short', 'Mid', 'High']
        year_bins = [0, 1940, 1980, float('inf')]
        year_labels = ['Historic (<1940)', 'Mid-Century (40-80)', 'Modern (>1980)']
        df_work['area_cat'] = pd.cut(df_work['Est GFA sqmeters'], bins=area_bins, labels=area_labels, right=False)
        df_work['height_cat'] = pd.cut(df_work['HEIGHT_USED'], bins=height_bins, labels=height_labels, right=False)
        df_work['year_cat'] = pd.cut(df_work['year_built'], bins=year_bins, labels=year_labels, right=False)
        hier = {}
        # All buildings
        hier['all'] = {
            'by_count': self._process_hierarchy(df_work, ['occ_cat','area_cat','height_cat','year_cat'], 'count'),
            'by_gfa':   self._process_hierarchy(df_work, ['occ_cat','area_cat','height_cat','year_cat'], 'gfa'),
            'by_count_simplified': self._process_hierarchy(df_work, ['occ_cat','year_cat'], 'count'),
            'by_gfa_simplified':   self._process_hierarchy(df_work, ['occ_cat','year_cat'], 'gfa'),
        }
        bin_info = {
            'Area': f"Small (<{area_bins[1]:.0f} sqm), Medium ({area_bins[1]:.0f}-{area_bins[2]:.0f} sqm), Large (>{area_bins[2]:.0f} sqm)",
            'Height': f"Short (<{height_bins[1]:.1f}m), Mid ({height_bins[1]:.1f}-{height_bins[2]:.1f}m), High (>{height_bins[2]:.1f}m)",
            'Year': f"Historic (<1940), Mid-Century (1940-1980), Modern (>1980)",
        }
        for v in hier['all']: hier['all'][v]['bin_info'] = bin_info
        # Per occupancy
        for oc in df_work['OCC_CLS'].unique():
            od = df_work[df_work['OCC_CLS'] == oc]
            if len(od) > 100:
                hier[oc] = {
                    'by_count': self._process_hierarchy(od, ['area_cat','height_cat','year_cat'], 'count', root_name=oc),
                    'by_gfa':   self._process_hierarchy(od, ['area_cat','height_cat','year_cat'], 'gfa', root_name=oc),
                    'by_count_simplified': self._process_hierarchy(od, ['year_cat'], 'count', root_name=oc),
                    'by_gfa_simplified':   self._process_hierarchy(od, ['year_cat'], 'gfa', root_name=oc),
                }
                for v in hier[oc]: hier[oc][v]['bin_info'] = bin_info
        print(f"  Hierarchical data for {len(hier)} groups"); return hier

    # ───────────────────── YEAR→OCC→MAT→FOUND flow (no soil) ─────────────────────
    def process_year_occ_mat_found_flow(self):
        """Year → Occupancy → Material → Foundation (no soil)."""
        print("Processing Year→Occ→Mat→Found flow …")
        df = self.df_cleaned.copy()
        year_bins = [0, 1940, 1980, float('inf')]
        year_labels = ['Historic (<1940)', 'Mid-Century (1940–1980)', 'Modern (>1980)']
        df['year_band'] = pd.cut(df['year_built'], bins=year_bins, labels=year_labels, right=False)
        df['occupancy'] = df['OCC_CLS'].fillna('Unknown') if 'OCC_CLS' in df.columns else 'Unknown'
        df['material']  = df['material_type'].fillna('Unknown') if 'material_type' in df.columns else 'Unknown'
        df['foundation'] = df['foundation_type'].fillna('Unknown') if 'foundation_type' in df.columns else 'Unknown'
        group_cols = ['year_band','occupancy','material','foundation']
        gc = df.groupby(group_cols, observed=True).size().reset_index(name='count')
        gg = df.groupby(group_cols, observed=True)['Est GFA sqmeters'].sum().reset_index(name='gfa') if 'Est GFA sqmeters' in df.columns else gc.copy()
        if 'gfa' not in gg.columns: gg['gfa'] = 0
        grp = gc.merge(gg, on=group_cols, how='left')
        grp['gfa'] = grp['gfa'].fillna(0).astype(float)
        records = []
        for _, r in grp.iterrows():
            records.append({c: (str(r[c]) if c != 'count' and c != 'gfa' else (int(r[c]) if c == 'count' else float(r[c]))) for c in group_cols + ['count','gfa']})
        return {'combination_counts': records, 'meta': {'year_order_top_to_bottom': year_labels, 'total_buildings': len(df)}}

    # ───────────────────── OCC_CLS → PRIM_OCC hierarchy ─────────────────────
    def process_occupancy_hierarchy(self):
        print("Processing OCC_CLS → PRIM_OCC hierarchy …")
        if 'PRIM_OCC' not in self.df_cleaned.columns or 'OCC_CLS' not in self.df_cleaned.columns:
            print("  Skipping: PRIM_OCC or OCC_CLS not found."); return None
        df = self.df_cleaned[['OCC_CLS','PRIM_OCC']].dropna()
        al = df.groupby(['OCC_CLS','PRIM_OCC']).size().reset_index(name='value')
        al = al[al['value'] > 0].copy()
        al['OCC_CLS_mod'] = al['OCC_CLS'].astype(str) + ' (Class)'
        cond = al['PRIM_OCC'] == 'Unclassified'
        al['PRIM_OCC_mod'] = np.where(cond, 'Unclassified (from ' + al['OCC_CLS'] + ')', al['PRIM_OCC'].astype(str) + ' (Type)')
        occ_nodes = al['OCC_CLS_mod'].unique().tolist()
        prim_nodes = al['PRIM_OCC_mod'].unique().tolist()
        all_nodes = occ_nodes + prim_nodes
        nm = {n: i for i, n in enumerate(all_nodes)}
        return {
            'nodes': [{'name': n} for n in all_nodes],
            'links': {'source': al['OCC_CLS_mod'].map(nm).tolist(), 'target': al['PRIM_OCC_mod'].map(nm).tolist(), 'value': al['value'].tolist()}
        }

    # ───────────────────── OCC_CLS → OCC_DICT sankey ─────────────────────
    def process_occ_cls_to_occdict_sankey(self):
        print("Processing OCC_CLS → NSI occtype (OCC_DICT) sankey …")
        if 'OCC_DICT' not in self.df_cleaned.columns or 'OCC_CLS' not in self.df_cleaned.columns:
            print("  Skipping: OCC_DICT or OCC_CLS not found."); return None
        df = self.df_cleaned[['OCC_CLS','OCC_DICT']].copy()
        def parse(v):
            if isinstance(v, dict): return v
            if pd.isna(v): return {}
            out = {}
            for p in str(v).strip().strip('{}').replace(';',',').split(','):
                if ':' in p:
                    k, val = p.split(':', 1)
                    try: out[k.strip()] = int(float(val.strip()))
                    except: pass
            return out
        rows = []
        for _, r in df.iterrows():
            occ = r['OCC_CLS']; d = parse(r['OCC_DICT'])
            for k, v in d.items():
                rows.append({'OCC_CLS': str(occ), 'occtype': str(k), 'points': int(v), 'has': 1 if int(v) > 0 else 0})
        if not rows: return None
        x = pd.DataFrame(rows)
        ap = x.groupby(['OCC_CLS','occtype'], observed=True)['points'].sum().reset_index()
        ap = ap[ap['points'] > 0].copy()
        def to_sankey(adf, vcol):
            left = adf['OCC_CLS'].astype(str) + ' (Class)'
            right = adf['occtype'].astype(str) + ' (from ' + adf['OCC_CLS'] + ')'
            nodes = pd.Index(pd.concat([left, right], ignore_index=True).unique())
            idx = {n: i for i, n in enumerate(nodes)}
            return {'nodes': [{'name': n} for n in nodes], 'links': {'source': [idx[s] for s in left], 'target': [idx[t] for t in right], 'value': adf[vcol].astype(float).tolist()}}
        return {'by_points': to_sankey(ap.rename(columns={'points':'value'}), 'value')}

    # ───────────────────── MIX_SC distribution ─────────────────────
    def process_mix_sc_distribution(self):
        print("Processing MIX_SC distribution …")
        if 'MIX_SC' not in self.df_cleaned.columns:
            print("  Skipping: MIX_SC not found.");
            return None

        counts = self.df_cleaned['MIX_SC'].value_counts(dropna=False)

        nan_count = int(self.df_cleaned['MIX_SC'].isna().sum())
        empty_str_count = int((self.df_cleaned['MIX_SC'] == '').sum())
        total_missing = nan_count + empty_str_count

        data = {
            'Same Type Only': total_missing,
            '1 Conflict Type (MIX_SC1)': int(counts.get('MIX_SC1', 0)),
            'Same & Different Types (MIX_SC2)': int(counts.get('MIX_SC2', 0)),
            '>1 Conflict Types (MIX_SC3)': int(counts.get('MIX_SC3', 0))
        }
        return {k: v for k, v in data.items() if v > 0}

    # ───────────────────── SAMPLES ─────────────────────
    def prepare_samples(self, n=75000):
        print(f"Preparing map samples (target {n:,}) …")
        if 'LATITUDE' not in self.df_cleaned.columns or 'LONGITUDE' not in self.df_cleaned.columns:
            print("  No coordinates."); return pd.DataFrame()
        vm = self.df_cleaned['LATITUDE'].notna() & self.df_cleaned['LONGITUDE'].notna() & (self.df_cleaned['LATITUDE'] != 0) & (self.df_cleaned['LONGITUDE'] != 0)
        dv = self.df_cleaned[vm]
        print(f"  Valid coords: {len(dv):,}")
        s = dv.sample(n=min(n, len(dv)), random_state=42)
        cols = ['LATITUDE','LONGITUDE']
        for c in ['OCC_CLS','year_built','Est GFA sqmeters','SQMETERS','HEIGHT_USED','material_type','foundation_type']:
            if c in s.columns: cols.append(c)
        return s[cols].copy()

    # ───────────────────── JSON CLEAN ─────────────────────
    def clean_for_json(self, obj):
        if isinstance(obj, dict): return {k: self.clean_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list): return [self.clean_for_json(v) for v in obj]
        if isinstance(obj, float):
            return None if (np.isnan(obj) or np.isinf(obj)) else obj
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)):
            return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
        if isinstance(obj, np.ndarray): return self.clean_for_json(obj.tolist())
        return obj

    # ───────────────────── EXPORT ─────────────────────
    def export_to_json(self, output_path='co_building_data.json'):
        print("Exporting to JSON …")
        occ_counts = self.get_overview_occupancy_counts()
        temporal = self.process_temporal_data()
        hier = self.process_hierarchical_distribution()
        year_occ_flow = self.process_year_occ_mat_found_flow()
        occ_hier = self.process_occupancy_hierarchy()
        occ_dict_sankey = self.process_occ_cls_to_occdict_sankey()
        mix_sc = self.process_mix_sc_distribution()
        sample_df = self.prepare_samples(n=75000)
        ss = {
            'total_buildings': len(self.df_cleaned),
            'avg_year_built': int(self.df_cleaned['year_built'].mean()) if 'year_built' in self.df_cleaned.columns else 0,
            'avg_area_sqm': float(self.df_cleaned['Est GFA sqmeters'].dropna().mean()) if 'Est GFA sqmeters' in self.df_cleaned.columns else 0,
            'min_year': int(self.df_cleaned['year_built'].min()) if 'year_built' in self.df_cleaned.columns else 0,
            'max_year': int(self.df_cleaned['year_built'].max()) if 'year_built' in self.df_cleaned.columns else 0,
            'occupancy_classes': sorted(self.df_cleaned['OCC_CLS'].unique().tolist()) if 'OCC_CLS' in self.df_cleaned.columns else []
        }
        samples_list = [self.clean_for_json(r) for r in sample_df.to_dict(orient='records')] if len(sample_df) else []
        main_data = {
            'metadata': {'total_buildings': len(self.df_cleaned), 'date_processed': datetime.now().isoformat(), 'source_file': self.file_path, 'version': '1.0-CO-overview', 'state': 'Colorado'},
            'summary_stats': ss,
            'overview_occupancy_counts': occ_counts,
            'temporal_data': temporal,
            'hierarchical_distribution': hier,
            'year_occ_flow': year_occ_flow,
            'occupancy_hierarchy_sankey': occ_hier,
            'occ_cls_occ_dict_sankey': occ_dict_sankey,
            'mix_sc_distribution': mix_sc,
            'building_samples_random': samples_list,
            'data_flow_stats': self.data_flow_stats
        }
        main_data = self.clean_for_json(main_data)
        # Split samples if big
        CHUNK = 5000; sfi = []
        if len(samples_list) > CHUNK:
            main_data['building_samples_random'] = []
            main_data['metadata']['has_samples_file'] = True
            main_data['metadata']['samples_split'] = True
            chunks = [samples_list[i:i+CHUNK] for i in range(0, len(samples_list), CHUNK)]
            for i, ch in enumerate(chunks):
                fn = output_path.replace('.json', f'_samples_random_{i+1}.json')
                cd = {'metadata': {'type':'random','chunk_index':i+1,'total_chunks':len(chunks),'chunk_size':len(ch),'date_generated':datetime.now().isoformat()}, 'samples': ch}
                with open(fn, 'w') as f: json.dump(cd, f, separators=(',',':'))
                mb = len(json.dumps(cd, separators=(',',':'))) / 1024 / 1024
                sfi.append({'filename': fn.split('/')[-1], 'type':'random', 'chunk_index':i+1, 'sample_count':len(ch), 'size_mb':round(mb,2)})
                print(f"  Saved {fn} ({mb:.2f} MB)")
            main_data['metadata']['samples_files'] = sfi
            main_data['metadata']['total_random_samples'] = len(samples_list)
        else:
            main_data['metadata']['has_samples_file'] = False
            main_data['metadata']['samples_split'] = False
        with open(output_path, 'w') as f: json.dump(main_data, f, indent=2)
        mb = len(json.dumps(main_data)) / 1024 / 1024
        print(f"\n{'='*60}\nExport Complete! → {output_path} ({mb:.2f} MB)\n{'='*60}")
        print(f"  Buildings: {ss['total_buildings']:,}  |  Occ classes: {len(occ_counts)}  |  Temporal pts: {len(temporal)}")
        print(f"  Hierarchical: {'Yes' if hier else 'No'}  |  Year→Occ flow: {'Yes' if year_occ_flow else 'No'}")
        print(f"  Occ hierarchy: {'Yes' if occ_hier else 'No'}  |  OCC_DICT sankey: {'Yes' if occ_dict_sankey else 'No'}  |  MIX_SC: {'Yes' if mix_sc else 'No'}")
        return main_data


def main():
    print("="*60); print("Colorado Building Data Processing – Overview Version"); print("="*60)
    p = COBuildingDataProcessor('usa_structures_nsi_with_height.gpkg')
    p.load_data(); p.clean_data()
    p.export_to_json('co_building_data.json')
    print("\nDone. Open co_index.html to visualize.")

if __name__ == "__main__":
    main()