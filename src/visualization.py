import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np
import sys, os, time, io, contextlib
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.grid_model import create_grid
from src.layout_validator import LayoutValidator
from src.astar_planner import astar
from src.fleet_selector import FleetSelector
from src.ml_pipeline import DemandForecaster, AnomalyDetector

DARK_BG   = "#0F1117"
CARD_BG   = "#1C1F2E"
ACCENT    = "#4F8EF7"
ACCENT2   = "#A78BFA"
SUCCESS   = "#22C55E"
DANGER    = "#EF4444"
WARNING   = "#F59E0B"

ZONE_COLORS = {
    "Residential": "#3B82F6",
    "Commercial":  "#F59E0B",
    "Industrial":  "#6B7280",
    "School":      "#10B981",
    "Hospital":    "#EC4899",
    "Open Field":  "#1E293B",
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] {{ font-family:'Inter',sans-serif; background:{DARK_BG}; color:#E2E8F0; }}
.stApp {{ background:{DARK_BG}; }}
section[data-testid="stSidebar"] {{ background:{CARD_BG}; border-right:1px solid #2D3748; }}
.block-container {{ padding:1.5rem 2rem; }}
h1 {{ background:linear-gradient(135deg,{ACCENT},{ACCENT2}); -webkit-background-clip:text;
       -webkit-text-fill-color:transparent; font-weight:700; font-size:2rem; margin-bottom:0; }}
.stTabs [data-baseweb="tab-list"] {{ background:{CARD_BG}; border-radius:12px; padding:4px; border:1px solid #2D3748; }}
.stTabs [data-baseweb="tab"] {{ color:#94A3B8; font-weight:500; border-radius:8px; padding:8px 20px; }}
.stTabs [aria-selected="true"] {{ background:{ACCENT}!important; color:#fff!important; }}
div[data-testid="metric-container"] {{ background:{CARD_BG}; border:1px solid #2D3748;
    border-radius:12px; padding:16px; }}
div[data-testid="metric-container"] label {{ color:#94A3B8; font-size:.75rem; text-transform:uppercase; letter-spacing:.08em; }}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {{ color:#E2E8F0; font-weight:700; font-size:1.6rem; }}
.stButton>button {{ background:linear-gradient(135deg,{ACCENT},{ACCENT2}); color:#fff;
    border:none; border-radius:8px; font-weight:600; padding:.5rem 1.5rem;
    transition:opacity .2s; }}
.stButton>button:hover {{ opacity:.85; }}
.stTextInput>div>div>input {{ background:#2D3748; border:1px solid #4A5568;
    border-radius:8px; color:#E2E8F0; }}
.stNumberInput>div>div>input {{ background:#2D3748; border:1px solid #4A5568;
    border-radius:8px; color:#E2E8F0; }}
.stCheckbox span {{ color:#CBD5E1; }}
.stAlert {{ border-radius:10px; }}
.stExpander {{ background:{CARD_BG}; border:1px solid #2D3748; border-radius:10px; }}
.legend-pill {{ display:inline-block; width:14px; height:14px; border-radius:4px; margin-right:6px; vertical-align:middle; }}
.log-line {{ font-family:monospace; font-size:.82rem; padding:3px 0; border-bottom:1px solid #1E293B; color:#CBD5E1; }}
</style>
"""

def draw_drone(ax, dr, dc):
    y, x = 9 - dr + 0.5, dc + 0.5
    ax.plot([x-.22,x+.22],[y-.22,y+.22], color='#111', lw=3.5, zorder=5)
    ax.plot([x-.22,x+.22],[y+.22,y-.22], color='#111', lw=3.5, zorder=5)
    for rx,ry in [(-0.22,-0.22),(-0.22,0.22),(0.22,-0.22),(0.22,0.22)]:
        ax.add_patch(patches.Circle((x+rx,y+ry),.10,color='#374151',zorder=6))
        ax.add_patch(patches.Circle((x+rx,y+ry),.04,color='#9CA3AF',zorder=7))
    ax.add_patch(patches.Circle((x,y),.15,color='#FCD34D',zorder=8))
    ax.add_patch(patches.Circle((x,y),.055,color='#EF4444',zorder=9))

def draw_grid(grid, path=None, old_path=None, disruption=None, drone_pos=None, mode="zones"):
    fig, ax = plt.subplots(figsize=(7,7))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#0F1117")
    ax.set_xlim(0,10); ax.set_ylim(0,10)

    if mode == "demand":
        demand_vals = np.array([[grid[r][c].demand for c in range(10)] for r in range(10)])
        norm = mcolors.Normalize(vmin=demand_vals.min(), vmax=max(demand_vals.max(),1))
        cmap = plt.cm.YlOrRd
        for r in range(10):
            for c in range(10):
                color = cmap(norm(demand_vals[r][c]))
                ax.add_patch(patches.Rectangle((c,9-r),1,1,lw=1,ec="#0F1117",fc=color))
                ax.text(c+.5,9-r+.5,f"{demand_vals[r][c]:.0f}",ha='center',va='center',
                        fontsize=7,color='white',fontweight='bold')
    else:
        for r in range(10):
            for c in range(10):
                cell = grid[r][c]; y,x = 9-r, c
                if cell.no_fly:
                    fc,label,fc_txt = "#1F2937","✗","#EF4444"
                elif cell.is_hub:
                    fc,label,fc_txt = "#1D4ED8","HUB","#93C5FD"
                elif cell.is_charging:
                    fc,label,fc_txt = "#065F46","CHG","#6EE7B7"
                else:
                    fc = ZONE_COLORS.get(cell.zone,"#1E293B")
                    label = cell.zone[0]
                    fc_txt = "#E2E8F0"
                ax.add_patch(patches.Rectangle((x,y),1,1,lw=1,ec="#0F1117",fc=fc))
                ax.text(x+.5,y+.5,label,ha='center',va='center',
                        fontsize=9,color=fc_txt,fontweight='bold')
    if old_path:
        px=[c+.5 for _,c in old_path]; py=[9-r+.5 for r,_ in old_path]
        ax.plot(px,py,color='#6B7280',lw=2,ls=':',alpha=.5)
    if path:
        px=[c+.5 for _,c in path]; py=[9-r+.5 for r,_ in path]
        ax.plot(px,py,color='#4F8EF7',lw=3,ls='--',marker='o',ms=5,zorder=3)
        ax.plot(px[0],py[0],color=SUCCESS,marker='s',ms=10,zorder=4)
        ax.plot(px[-1],py[-1],color=ACCENT2,marker='*',ms=14,zorder=4)
    if disruption:
        dr,dc=disruption
        ax.plot(dc+.5,9-dr+.5,color=WARNING,marker='X',ms=18,mec='black',zorder=4)
    if drone_pos:
        draw_drone(ax,drone_pos[0],drone_pos[1])
    ax.axis('off')
    return fig

def generate_frames(grid, start, goal, simulate_disruption):
    frames=[]; path,cost,msg = astar(start,goal,grid)
    if not path: return None,msg
    fpf=8
    if simulate_disruption and len(path)>3:
        didx=len(path)//2; dcell=path[didx]; old=list(path)
        for i in range(didx-1):
            r1,c1=path[i]; r2,c2=path[i+1]
            for s in range(fpf):
                frames.append({'pos':(r1+(r2-r1)*s/fpf,c1+(c2-c1)*s/fpf),
                    'path':path[:didx],'old_path':None,'disruption':None,
                    'msg':"🚁 Flight initiated — navigating to target..."})
        grid[dcell[0]][dcell[1]].no_fly=True
        nloc=path[didx-1]; np2,_,nm=astar(nloc,goal,grid)
        grid[dcell[0]][dcell[1]].no_fly=False
        if np2:
            merged=path[:didx]+np2[1:]
            for i in range(didx-1,len(merged)-1):
                r1,c1=merged[i]; r2,c2=merged[i+1]
                for s in range(fpf):
                    frames.append({'pos':(r1+(r2-r1)*s/fpf,c1+(c2-c1)*s/fpf),
                        'path':merged,'old_path':old,'disruption':dcell,
                        'msg':f"💥 No-fly at {dcell}! A* rerouted successfully."})
            frames.append({'pos':merged[-1],'path':merged,'old_path':old,
                'disruption':dcell,'msg':"🎉 Delivery complete (rerouted)!"})
            return frames,"Success"
        return None,"Reroute failed — drone trapped."
    for i in range(len(path)-1):
        r1,c1=path[i]; r2,c2=path[i+1]
        for s in range(fpf):
            frames.append({'pos':(r1+(r2-r1)*s/fpf,c1+(c2-c1)*s/fpf),
                'path':path,'old_path':None,'disruption':None,
                'msg':"🚁 Navigating smoothly..."})
    frames.append({'pos':path[-1],'path':path,'old_path':None,
        'disruption':None,'msg':"🎉 Delivery complete!"})
    return frames,"Success"

def run_capture(fn):
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):
        result=fn()
    return result, buf.getvalue()

def main():
    st.set_page_config(page_title="AeroNet Lite", page_icon="🚁", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown("# 🚁 AeroNet Lite")
    st.markdown("<p style='color:#64748B;margin-top:-.5rem;'>Autonomous Drone Delivery Simulation · AI Module SP2026</p>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Sidebar ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Grid Controls")
        fix_layout = st.checkbox("🔧 Fix all CSP violations")
        trap_drone = st.checkbox("🧱 Trap drone at (0,0)")
        st.markdown("---")
        st.markdown("### 💰 Fleet Budget")
        budget_input  = st.number_input("Budget ($)", value=10000, step=500)
        demand_input  = st.number_input("Target Demand", value=50, step=5)
        st.markdown("---")
        st.markdown("### 🗺️ Map Mode")
        map_mode = st.radio("View", ["Zone Map","Demand Heatmap"], label_visibility="collapsed")

    grid = create_grid()

    # Apply demand values for heatmap
    for r in range(10):
        for c in range(10):
            grid[r][c].demand = grid[r][c].density / 100.0

    if fix_layout:
        for r in range(10):
            for c in range(10):
                grid[r][c].is_hub=True; grid[r][c].is_charging=True
    if trap_drone:
        grid[0][1].no_fly=True; grid[1][0].no_fly=True

    # ── KPI Row ──────────────────────────────────────────────────
    validator = LayoutValidator(grid)
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf): is_valid=validator.run_validation()

    hub_count    = sum(1 for r in range(10) for c in range(10) if grid[r][c].is_hub)
    hosp_count   = sum(1 for r in range(10) for c in range(10) if grid[r][c].zone=="Hospital")
    nf_count     = sum(1 for r in range(10) for c in range(10) if grid[r][c].no_fly)
    rule_errors  = len(validator.errors)

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Grid Size", "10 × 10")
    k2.metric("Drone Hubs", hub_count)
    k3.metric("Hospitals", hosp_count)
    k4.metric("No-Fly Zones", nf_count)
    k5.metric("CSP Violations", rule_errors, delta=f"{rule_errors} rules failed" if rule_errors else "All passed", delta_color="inverse")
    st.markdown("---")

    # ── Tabs ─────────────────────────────────────────────────────
    t1,t2,t3,t4,t5 = st.tabs(["🗺️ Flight Planner","🔍 Layout Validator","🧬 Fleet Selector","📈 ML Forecast","⚠️ Anomaly Detector"])

    # ── TAB 1: Flight Planner ────────────────────────────────────
    with t1:
        col1,col2 = st.columns([3,1])
        with col2:
            st.markdown("#### Route Controls")
            start_point = st.text_input("Start (row,col)","1,3")
            goal_point  = st.text_input("Goal  (row,col)","7,5")
            sync_hub    = st.checkbox("Pin start as Hub")
            trigger_dis = st.checkbox("💥 Trigger Disruption")
            calc_btn    = st.button("🚀 Fly Drone", use_container_width=True)
            st.markdown("---")
            st.markdown("**Legend**")
            for zone,color in ZONE_COLORS.items():
                st.markdown(f"<span class='legend-pill' style='background:{color}'></span>{zone}", unsafe_allow_html=True)
            st.markdown(f"<span class='legend-pill' style='background:#1D4ED8'></span>Hub", unsafe_allow_html=True)
            st.markdown(f"<span class='legend-pill' style='background:#065F46'></span>Charging", unsafe_allow_html=True)
            st.markdown(f"<span class='legend-pill' style='background:#1F2937'></span>No-Fly ✗", unsafe_allow_html=True)

        with col1:
            try:
                sr,sc=map(int,start_point.split(','))
                gr,gc=map(int,goal_point.split(','))
                if sync_hub:
                    grid[sr][sc].is_hub=True; grid[sr][sc].zone='Commercial'
            except:
                sr,sc,gr,gc=1,3,7,5

            status_box   = st.empty()
            plot_box     = st.empty()

            if calc_btn:
                frames,msg = generate_frames(grid,(sr,sc),(gr,gc),trigger_dis)
                if frames:
                    st.session_state['frames']=frames
                else:
                    st.session_state.pop('frames',None)
                    status_box.error(f"❌ Routing failed: {msg}")

            if 'frames' in st.session_state and st.session_state['frames']:
                for frame in st.session_state['frames']:
                    m=frame['msg']
                    if "💥" in m or "ALERT" in m: status_box.warning(m)
                    elif "🎉" in m: status_box.success(m)
                    else: status_box.info(m)
                    mode = "demand" if map_mode=="Demand Heatmap" else "zones"
                    fig=draw_grid(grid,path=frame['path'],old_path=frame['old_path'],
                        disruption=frame['disruption'],drone_pos=frame['pos'],mode=mode)
                    plot_box.pyplot(fig); plt.close(fig)
                    time.sleep(0.025)
                st.session_state.pop('frames',None)
            else:
                mode="demand" if map_mode=="Demand Heatmap" else "zones"
                plot_box.pyplot(draw_grid(grid,mode=mode))

    # ── TAB 2: Layout Validator ──────────────────────────────────
    with t2:
        st.markdown("#### CSP Constraint Checker")
        c1,c2 = st.columns([1,1])
        with c1:
            if is_valid:
                st.success("✅ Grid layout is fully valid — all 4 rules pass.")
            else:
                st.warning(f"⚠️ {rule_errors} constraint violation(s) detected.")
            st.markdown("**Rules evaluated:**")
            rule_defs = [
                ("R1","Industrial Safety","No industrial cell adjacent to School/Hospital"),
                ("R2","Residential Coverage","All residential zones ≤ 3 cells from a Hub"),
                ("R3","Hub Charging Access","Every Hub has a Charging Pad within distance 2"),
                ("R4","Medical Access","All Hospitals marked as pickup & near a Hub"),
            ]
            passed = [r for r in validator.passed_rules]
            for rid,name,desc in rule_defs:
                ok = any(rid in p for p in passed)
                icon = "✅" if ok else "❌"
                st.markdown(f"{icon} **{rid} — {name}**: {desc}")
        with c2:
            if validator.errors:
                st.markdown("**Violations & Fixes:**")
                for err in validator.errors:
                    st.error(err)
            else:
                st.info("No violations found.")
        st.markdown("---")
        st.pyplot(draw_grid(grid,mode="zones"))

    # ── TAB 3: Fleet Selector ────────────────────────────────────
    with t3:
        st.markdown("#### Genetic Algorithm Fleet Optimizer")
        c1,c2=st.columns([1,1])
        with c1:
            st.markdown("""
| Drone | Cost | Payload | Range |
|-------|------|---------|-------|
| Light 🟡 | $1,000 | 5 kg | 12 cells |
| Heavy 🔵 | $2,500 | 15 kg | 20 cells |

**Fitness:** `score = 0.75 × coverage% − 0.25 × budget_used%`
""")
            gens = st.slider("Generations", 10,100,50,10)
            pop  = st.slider("Population Size", 10,50,20,5)
            run_ga = st.button("🧬 Evolve Fleet", use_container_width=True)
        with c2:
            if run_ga:
                with st.spinner("Evolving optimal fleet..."):
                    sel=FleetSelector(total_demand=int(demand_input),budget=int(budget_input))
                    fleet,score=sel.run_genetic_algorithm(generations=gens,pop_size=pop)
                nl,nh=fleet
                cost=nl*1000+nh*2500
                cap=nl*5+nh*15
                st.success(f"**Best Fleet: {nl} Light + {nh} Heavy Drones**")
                m1,m2,m3,m4=st.columns(4)
                m1.metric("Light Drones",nl)
                m2.metric("Heavy Drones",nh)
                m3.metric("Total Cost",f"${cost:,}")
                m4.metric("Capacity",f"{cap} units")
                used_pct=cost/budget_input*100
                cov_pct=min(100,cap/demand_input*100)
                st.progress(int(used_pct),text=f"Budget used: {used_pct:.1f}%")
                st.progress(int(cov_pct), text=f"Demand covered: {cov_pct:.1f}%")
                st.metric("Fitness Score", f"{score:.4f}")
            else:
                st.info("Set budget & demand in the sidebar, then click **Evolve Fleet**.")

    # ── TAB 4: Demand Forecast ───────────────────────────────────
    with t4:
        st.markdown("#### Demand Forecasting — Random Forest Regressor")
        st.markdown("Trained on the **Kaggle Bike Sharing Demand** dataset using weather & calendar features.")
        run_reg=st.button("📈 Train & Forecast", use_container_width=False)
        if run_reg:
            data_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data","raw","train.csv")
            forecaster=DemandForecaster(data_path)
            with st.spinner("Training Random Forest Regressor..."):
                result,out=run_capture(forecaster.run)
            mae_lines=[l for l in out.splitlines() if "MAE" in l]
            pred_lines=[l for l in out.splitlines() if "Sample" in l]
            c1,c2,c3=st.columns(3)
            c1.metric("Model","Random Forest")
            if mae_lines: c2.metric("MAE",mae_lines[0].split(":")[-1].strip())
            if result: c3.metric("Avg Predicted Demand",f"{result:.0f} units")
            if pred_lines: st.info(pred_lines[0])
            with st.expander("Full training output"):
                st.code(out)

            # Feature importance chart
            if hasattr(forecaster.model,'feature_importances_'):
                feats=['season','holiday','workingday','weather','temp','humidity','windspeed']
                imps=forecaster.model.feature_importances_
                fig,ax=plt.subplots(figsize=(7,3))
                fig.patch.set_facecolor(CARD_BG)
                ax.set_facecolor(CARD_BG)
                bars=ax.barh(feats,imps,color=ACCENT)
                ax.set_xlabel("Importance",color="#94A3B8")
                ax.tick_params(colors="#CBD5E1")
                ax.spines[:].set_color("#2D3748")
                ax.set_title("Feature Importances",color="#E2E8F0",fontweight='bold')
                st.pyplot(fig)
        else:
            st.info("Click **Train & Forecast** to train the model on real data.")

    # ── TAB 5: Anomaly Detector ──────────────────────────────────
    with t5:
        st.markdown("#### Anomaly Detection — Random Forest Classifier")
        st.markdown("Trained on **2,000 synthetic drone telemetry samples** (15% anomaly rate).")
        run_cls=st.button("⚠️ Train Classifier", use_container_width=False)
        if run_cls:
            detector=AnomalyDetector()
            with st.spinner("Training classifier on synthetic telemetry..."):
                result,out=run_capture(detector.run)
            acc,cm=result
            c1,c2,c3=st.columns(3)
            c1.metric("Model","Random Forest")
            c2.metric("Accuracy",f"{acc*100:.2f}%")
            c3.metric("Anomaly Rate","~15%")

            # Confusion matrix heatmap
            fig,ax=plt.subplots(figsize=(4,3))
            fig.patch.set_facecolor(CARD_BG)
            ax.set_facecolor(CARD_BG)
            im=ax.imshow(cm,cmap='Blues')
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(["Normal","Anomaly"],color="#CBD5E1")
            ax.set_yticklabels(["Normal","Anomaly"],color="#CBD5E1")
            ax.set_xlabel("Predicted",color="#94A3B8")
            ax.set_ylabel("Actual",color="#94A3B8")
            ax.set_title("Confusion Matrix",color="#E2E8F0",fontweight='bold')
            for i in range(2):
                for j in range(2):
                    ax.text(j,i,cm[i,j],ha='center',va='center',
                            color='white' if cm[i,j]>cm.max()/2 else '#1E293B',fontweight='bold',fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)

            # Live classify custom sample
            st.markdown("---")
            st.markdown("**🔬 Classify a Custom Telemetry Reading**")
            sc1,sc2,sc3=st.columns(3)
            batt = sc1.slider("Battery Drop (%/sector)",0.0,35.0,5.0,.5)
            spd  = sc2.slider("Speed (m/s)",0.0,30.0,15.0,.5)
            dev  = sc3.slider("Route Deviation (m)",0.0,100.0,2.0,1.0)
            sample=pd.DataFrame([[batt,spd,dev]],columns=['battery_drop','speed','route_deviation'])
            pred=detector.model.predict(sample)[0]
            prob=detector.model.predict_proba(sample)[0]
            if pred==1:
                st.error(f"🚨 **ANOMALY DETECTED** — Confidence: {prob[1]*100:.1f}%")
            else:
                st.success(f"✅ **Normal Flight** — Confidence: {prob[0]*100:.1f}%")
        else:
            st.info("Click **Train Classifier** to train on synthetic telemetry data.")

if __name__ == "__main__":
    main()
