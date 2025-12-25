For some reasons entry and exit time are similar:
2	2019.05.24 01:16	2019.05.24 01:16	-15	30.5	22.5
4	2019.05.24 07:32	2019.05.24 07:32	-30.5	18	-30.5
6	2019.05.24 13:29	2019.05.24 13:29	-6.5	45	15.5
9	2019.05.24 16:40	2019.05.24 16:40	-63	4	-63
13	2019.05.26 23:11	2019.05.26 23:11	-10	9	-10
16	2019.05.27 00:40	2019.05.27 00:40	-31.5	57	43.5
20	2019.05.27 09:42	2019.05.27 09:42	-18	5	-18
22	2019.05.27 11:27	2019.05.27 11:27	-12	11	-6

Please be kind check the code for me:
//+------------------------------------------------------------------+
//| Red-Green Breakout EA: MAE/MFE via floating PnL (robust version)  |
//+------------------------------------------------------------------+
#property strict

input double Lots           = 1.0;
input double RiskReward     = 1.0;
input int    Slippage       = 5;

// flattening
input bool   UseFlatten     = true;
input int    FlattenHour    = 23;
input int    FlattenMinute  = 30;

// no-trading window
input bool   UseNoTradeWindow   = true;
input int    NoTradeStartHour   = 23;
input int    NoTradeStartMinute = 30;
input int    NoTradeEndHour     = 1;
input int    NoTradeEndMinute   = 0;

// ---- MAE / MFE (FLOATING PNL BASED) ----
double g_maeMoney   = 0.0;   // most negative floating PnL
double g_mfeMoney   = 0.0;   // most positive floating PnL
bool   g_tracking   = false;
ulong  g_ticket     = 0;

datetime g_entryTime = 0;
string   g_csvName   = "trade_stats.csv";

//---------------- Helpers ----------------//
bool IsFlattenTime(datetime barOpen)
{
   MqlDateTime dt; TimeToStruct(barOpen, dt);
   return (dt.hour == FlattenHour && dt.min == FlattenMinute);
}

bool InNoTradeWindow(datetime barOpen)
{
   if(!UseNoTradeWindow) return false;
   MqlDateTime dt; TimeToStruct(barOpen, dt);
   int cur = dt.hour * 60 + dt.min;
   int st  = NoTradeStartHour * 60 + NoTradeStartMinute;
   int en  = NoTradeEndHour   * 60 + NoTradeEndMinute;

   if(st <= en) return (cur >= st && cur < en);
   return (cur >= st || cur < en);
}

void CancelAllOrders()
{
   for(int i=OrdersTotal()-1;i>=0;i--)
   {
      ulong t = OrderGetTicket(i);
      if(!OrderSelect(t)) continue;

      MqlTradeRequest r={};
      MqlTradeResult  s={};
      r.action = TRADE_ACTION_REMOVE;
      r.order  = t;
      if(!OrderSend(r,s))
        Print("❌ CancelAllOrders failed ticket=", t, " err=", GetLastError());
      else
        Print("✅ Cancelled order ticket=", t);

   }
}

void CloseAllPositions()
{
   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      ulong t = PositionGetTicket(i);
      if(!PositionSelectByTicket(t)) continue;

      MqlTradeRequest r={};
      MqlTradeResult  s={};

      r.action    = TRADE_ACTION_DEAL;
      r.symbol    = _Symbol;
      r.volume    = PositionGetDouble(POSITION_VOLUME);
      r.deviation = Slippage;

      if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY)
      {
         r.type  = ORDER_TYPE_SELL;
         r.price = SymbolInfoDouble(_Symbol,SYMBOL_BID);
      }
      else
      {
         r.type  = ORDER_TYPE_BUY;
         r.price = SymbolInfoDouble(_Symbol,SYMBOL_ASK);
      }

      if(!OrderSend(r,s))
        Print("❌ CloseAllPositions failed pos#", t, " err=", GetLastError());
      else
        Print("✅ Closed position #", t);

   }
}

void CancelOldBuyStops()
{
   for(int i=OrdersTotal()-1;i>=0;i--)
   {
      ulong t = OrderGetTicket(i);
      if(!OrderSelect(t)) continue;
      if(OrderGetInteger(ORDER_TYPE)!=ORDER_TYPE_BUY_STOP) continue;

      MqlTradeRequest r={};
      MqlTradeResult  s={};
      r.action = TRADE_ACTION_REMOVE;
      r.order  = t;

      if(!OrderSend(r,s))
        Print("❌ Failed to cancel BuyStop ticket=", t, " err=", GetLastError());
      else
        Print("✅ Cancelled BuyStop ticket=", t);

   }
}

//---------------- CSV ----------------//
void SaveTradeStats(double realized, datetime entryTime, datetime exitTime)
{
   int f = FileOpen(g_csvName, FILE_READ|FILE_WRITE|FILE_CSV|FILE_SHARE_WRITE);
   if(f==INVALID_HANDLE){ Print("File open failed ",GetLastError()); return; }

   if(FileSize(f)==0)
      FileWrite(f,"ticket","entry_time","exit_time","mae_money","mfe_money","trade_profit");

   FileSeek(f,0,SEEK_END);
   FileWrite(
      f,
      (long)g_ticket,
      TimeToString(entryTime, TIME_DATE|TIME_MINUTES),
      TimeToString(exitTime,  TIME_DATE|TIME_MINUTES),
      g_maeMoney,
      g_mfeMoney,
      realized
   );

   FileClose(f);
}



//---------------- Trade mgmt ----------------//
void ManageOpenPosition()
{
   if(!PositionSelect(_Symbol)) return;

   double entry = PositionGetDouble(POSITION_PRICE_OPEN);
   double sl    = PositionGetDouble(POSITION_SL);
   double vol   = PositionGetDouble(POSITION_VOLUME);

   double risk = entry - sl;
   if(risk<=0) return;

   double barClose = iClose(_Symbol,_Period,1);

   if(barClose >= entry + risk*RiskReward)
   {
      Print("✅ ≥ ", RiskReward, "R at bar close → closing at market");
      MqlTradeRequest r={};
      MqlTradeResult  s={};

      r.action    = TRADE_ACTION_DEAL;
      r.symbol    = _Symbol;
      r.volume    = vol;
      r.type      = ORDER_TYPE_SELL;
      r.price     = SymbolInfoDouble(_Symbol,SYMBOL_BID);
      r.deviation = Slippage;

      if(!OrderSend(r,s))
         Print("❌ Close fail err=", GetLastError());
      else
         Print("✅ Position closed");
   }
   else
   {
       Print("⏳ Not yet ", RiskReward, "R on close → hold");
   }
}

//---------------- EA Core ----------------//
int OnInit(){ return INIT_SUCCEEDED; }

void OnTick()
{
   datetime barOpen = iTime(_Symbol, _Period, 0);

   // ---- TRACK FLOATING MAE / MFE (tick-based, broker-safe) ----
   if(PositionSelect(_Symbol))
   {
      double floating = PositionGetDouble(POSITION_PROFIT);

      if(!g_tracking)
      {
         g_tracking   = true;
         g_ticket     = PositionGetInteger(POSITION_TICKET);
         g_entryTime  = (datetime)PositionGetInteger(POSITION_TIME);
         g_maeMoney   = floating;
         g_mfeMoney   = floating;
         Print("Tracking started ticket=",g_ticket);
      }

      g_maeMoney = MathMin(g_maeMoney, floating);
      g_mfeMoney = MathMax(g_mfeMoney, floating);
   }
   else if(g_tracking)
   {
   // --- include final realized PnL into MAE/MFE ---
   double realized = 0.0;

   // --- include datetime ---
   datetime exitTime = 0;

   if(HistorySelect(g_entryTime - 86400, TimeCurrent()))
   {
      for(int i=HistoryDealsTotal()-1;i>=0;i--)
      {
         ulong deal = HistoryDealGetTicket(i);
         if(HistoryDealGetInteger(deal, DEAL_POSITION_ID) == g_ticket)
            {
				realized += HistoryDealGetDouble(deal, DEAL_PROFIT);
				exitTime  = (datetime)HistoryDealGetInteger(deal, DEAL_TIME);
			}
      }
   }

   // realized PnL IS the last excursion
   g_maeMoney = MathMin(g_maeMoney, realized);
   g_mfeMoney = MathMax(g_mfeMoney, realized);

   SaveTradeStats(realized, g_entryTime, exitTime);

   g_tracking = false;
   }

   // ---- BAR-GATED LOGIC ----
   static datetime lastBar=0;
   if(barOpen==lastBar) return;
   lastBar=barOpen;

   // 🔹 flatten
   if(UseFlatten && IsFlattenTime(barOpen))
   {
      Print("🌙 Flatten cutoff reached → closing everything");
      CloseAllPositions();
      CancelAllOrders();
      return;
   }

   // 🔹 no-trade window
   if(InNoTradeWindow(barOpen))
   {
      Print("🚫 In no-trading window → flat only, no new trades");
      CloseAllPositions();
      CancelAllOrders();
      return;
   }

   // 🔹 manage trade (bar-based logic)
   if(PositionsTotal() > 0)
   {
      ManageOpenPosition();
      return;
   }

   // 🔹 setup (unchanged)
   double o = iOpen (_Symbol, _Period, 1);
   double h = iHigh (_Symbol, _Period, 1);
   double l = iLow  (_Symbol, _Period, 1);
   double c = iClose(_Symbol, _Period, 1);

   if(c < o)
   {
      Print("🔴 Red candle → refresh BuyStop");
      CancelOldBuyStops();

      double risk = h - l;
      if(risk <= 0) return;

      MqlTradeRequest r = {};
      MqlTradeResult  s = {};

      r.action       = TRADE_ACTION_PENDING;
      r.symbol       = _Symbol;
      r.volume       = Lots;
      r.type         = ORDER_TYPE_BUY_STOP;
      r.price        = h;
      r.sl           = l;
      r.deviation    = Slippage;
      r.type_filling = ORDER_FILLING_RETURN;

      if(!OrderSend(r, s))
         Print("❌ Place BuyStop fail err=", GetLastError());
      else
         Print("🚀 BuyStop placed @", h, " SL=", l);
   }
}