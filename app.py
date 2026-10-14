import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title='BIST Teknik Analiz',layout='wide')
st.title('BIST Teknik Analiz | 1–7 Gün')
st.caption('Eğitim amaçlıdır. Veriler gecikmeli veya eksik olabilir; yatırım tavsiyesi değildir.')
symbol=st.text_input('Hisse kodu (ör. ALARK, ALBTN, METEN)', 'ALARK').strip().upper().replace('.IS','')
period=st.selectbox('Veri aralığı',['6mo','3mo','1y'],index=0)

def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def rsi(s,n=14):
    delta=s.diff(); up=delta.clip(lower=0); down=-delta.clip(upper=0)
    avg_up=up.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    avg_down=down.ewm(alpha=1/n,adjust=False,min_periods=n).mean()
    rs=avg_up/avg_down.replace(0,np.nan)
    result=100-100/(1+rs)
    return result.mask((avg_down==0)&(avg_up>0),100).mask((avg_down==0)&(avg_up==0),50)

@st.cache_data(ttl=900)
def load(sym,period):
    x=yf.download(sym+'.IS',period=period,interval='1d',auto_adjust=True,progress=False)
    if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
    return x.dropna(subset=['Close']).copy()

if st.button('Analiz et',type='primary'):
    try:
        df=load(symbol,period)
        if len(df)<65: st.error('Yeterli fiyat verisi bulunamadı (en az 65 günlük veri gerekir).'); st.stop()
        c=df['Close']; df['EMA9']=ema(c,9); df['EMA21']=ema(c,21)
        df['MACD']=ema(c,12)-ema(c,26); df['Signal']=ema(df['MACD'],9); df['Hist']=df['MACD']-df['Signal']
        df['RSI']=rsi(c); df['Vol20']=df['Volume'].rolling(20).mean()
        df['BBmid']=c.rolling(20).mean(); std=c.rolling(20).std(); df['BBupper']=df['BBmid']+2*std; df['BBlower']=df['BBmid']-2*std
        prevclose=c.shift(1); tr=pd.concat([(df['High']-df['Low']),(df['High']-prevclose).abs(),(df['Low']-prevclose).abs()],axis=1).max(axis=1)
        df['ATR']=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
        # Resistance/support use prior bars only, excluding the current bar.
        df['Resistance20']=df['High'].shift(1).rolling(20).max(); df['Support20']=df['Low'].shift(1).rolling(20).min()
        df['Resistance60']=df['High'].shift(1).rolling(60).max(); df['Support60']=df['Low'].shift(1).rolling(60).min()
        a=df.iloc[-1]; b=df.iloc[-2]
        bull_cross=b.MACD<=b.Signal and a.MACD>a.Signal
        bear_cross=b.MACD>=b.Signal and a.MACD<a.Signal
        breakout=a.Close>a.Resistance20 and a.Volume>1.5*a.Vol20
        st.subheader(f'{symbol} • {df.index[-1].date()}')
        cols=st.columns(5)
        for col,(label,val) in zip(cols,[('Kapanış',f'{a.Close:.2f} TL'),('RSI 14',f'{a.RSI:.1f}'),('MACD',f'{a.MACD:.3f}'),('Hacim / 20G',f'{a.Volume/a.Vol20:.2f}x'),('ATR 14',f'{a.ATR:.2f} TL')]): col.metric(label,val)
        notes=[f'MACD: {"bugün yukarı kesişim" if bull_cross else "bugün aşağı kesişim" if bear_cross else "bugün yeni kesişim yok"}; sıfırın {"üzerinde" if a.MACD>0 else "altında"}.',f'RSI: {a.RSI:.1f} ({"70 üzeri" if a.RSI>=70 else "30 altı" if a.RSI<=30 else "50 üzeri" if a.RSI>=50 else "30–50 arası"}).',f'EMA: EMA9 {"EMA21 üzerinde" if a.EMA9>a.EMA21 else "EMA21 altında"}.',f'Hacim: 20 günlük ortalamanın {a.Volume/a.Vol20:.2f} katı.',f'20G destek/direnç: {a.Support20:.2f} / {a.Resistance20:.2f} TL.',f'60G destek/direnç: {a.Support60:.2f} / {a.Resistance60:.2f} TL.',f'20G direnç üstünde hacimli kapanış: {"Evet" if breakout else "Hayır"}.']
        st.info('\n\n'.join(notes))
        fig=make_subplots(rows=4,cols=1,shared_xaxes=True,vertical_spacing=.035,row_heights=[.46,.16,.20,.18],subplot_titles=['Fiyat / EMA / Bollinger / Destek-Direnç','Hacim','MACD / Sinyal / Histogram','RSI'])
        fig.add_trace(go.Candlestick(x=df.index,open=df.Open,high=df.High,low=df.Low,close=df.Close,name='Mum'),row=1,col=1)
        for key,color in [('EMA9','#2563eb'),('EMA21','#f59e0b'),('BBupper','#9ca3af'),('BBlower','#9ca3af'),('Resistance20','#dc2626'),('Support20','#16a34a')]:
            fig.add_trace(go.Scatter(x=df.index,y=df[key],name=key,line=dict(color=color,width=1.5,dash='dot' if '20' in key and key not in ['EMA21'] else 'solid')),row=1,col=1)
        fig.add_trace(go.Bar(x=df.index,y=df.Volume,name='Hacim',marker_color='#94a3b8'),row=2,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df.MACD,name='MACD',line_color='#2563eb'),row=3,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df.Signal,name='Sinyal',line_color='#f59e0b'),row=3,col=1)
        fig.add_trace(go.Bar(x=df.index,y=df.Hist,name='Histogram',marker_color=np.where(df.Hist>=0,'#16a34a','#dc2626')),row=3,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df.RSI,name='RSI',line_color='#8b5cf6'),row=4,col=1)
        for level in (30,50,70): fig.add_hline(y=level,row=4,col=1,line_dash='dot',line_color='gray')
        fig.update_layout(height=950,xaxis_rangeslider_visible=False,legend=dict(orientation='h'),margin=dict(l=30,r=20,t=50,b=20))
        st.plotly_chart(fig,use_container_width=True)
        st.subheader('Risk / getiri senaryosu')
        st.caption('Bu seviyeler otomatik alım-satım tavsiyesi değildir; varsayımsal hesaplama aracıdır.')
        buy=st.number_input('Varsayımsal giriş fiyatı',min_value=.01,value=float(round(a.Close,2)))
        stop=st.number_input('Varsayımsal stop',min_value=.01,value=float(round(max(.01,buy-1.5*a.ATR),2)))
        target=st.number_input('Varsayımsal hedef',min_value=.01,value=float(round(buy+3*a.ATR,2)))
        if stop<buy<target: st.write(f'Risk/getiri: **1 : {(target-buy)/(buy-stop):.2f}** | Risk: **{(buy-stop)/buy*100:.2f}%** | Hedef: **{(target-buy)/buy*100:.2f}%**')
        else: st.warning('Stop < giriş < hedef olmalı.')
        st.download_button('Hesaplanan verileri CSV indir',df.to_csv().encode('utf-8-sig'),file_name=f'{symbol}_teknik_analiz.csv',mime='text/csv')
    except Exception as ex:
        st.error(f'Veri alınamadı: {ex}. Hisse kodunu ve veri sağlayıcısını kontrol et.')
