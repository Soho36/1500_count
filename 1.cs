//+------------------------------------------------------------------+
//| Red-Green Breakout EA: MAE/MFE via floating PnL (robust version)  |
//+------------------------------------------------------------------+
#property strict

input double Lots           = 1.0;
input double RiskReward     = 1.0;
input int    Slippage       = 5;

// flattening during session
input bool   UseFlattenDur     = true;
input int    FlattenHourDur    = 14;
input int    FlattenMinuteDur  = 00;

// flattening end session
input bool   UseFlattenEnd     = true;
input int    FlattenHourEnd    = 20;
input int    FlattenMinuteEnd  = 00;

// ---- 30-min Trading Windows (48 slots) ----
input bool UseTradeWindow = true;

input bool W00 = false;  // 00:00–00:30
input bool W01 = false;  // 00:30–01:00
input bool W02 = false;  // 01:00–01:30
input bool W03 = false;  // 01:30–02:00
input bool W04 = false;  // 02:00–02:30
input bool W05 = false;  // 02:30–03:00
input bool W06 = false;  // 03:00–03:30
input bool W07 = false;  // 03:30–04:00
input bool W08 = false;  // 04:00–04:30
input bool W09 = false;  // 04:30–05:00
input bool W10 = false;  // 05:00–05:30
input bool W11 = false;  // 05:30–06:00
input bool W12 = false;  // 06:00–06:30
input bool W13 = false;  // 06:30–07:00
input bool W14 = false;  // 07:00–07:30
input bool W15 = false;  // 07:30–08:00
input bool W16 = false;  // 08:00–08:30
input bool W17 = false;  // 08:30–09:00
input bool W18 = false;  // 09:00–09:30
input bool W19 = false;  // 09:30–10:00
input bool W20 = false;  // 10:00–10:30
input bool W21 = false;  // 10:30–11:00
input bool W22 = false;  // 11:00–11:30
input bool W23 = false;  // 11:30–12:00
input bool W24 = false;  // 12:00–12:30
input bool W25 = false;  // 12:30–13:00
input bool W26 = false;  // 13:00–13:30
input bool W27 = false;  // 13:30–14:00
input bool W28 = false;  // 14:00–14:30
input bool W29 = false;  // 14:30–15:00
input bool W30 = false;  // 15:00–15:30
input bool W31 = false;  // 15:30–16:00
input bool W32 = false;  // 16:00–16:30
input bool W33 = false;  // 16:30–17:00
input bool W34 = false;  // 17:00–17:30
input bool W35 = false;  // 17:30–18:00
input bool W36 = false;  // 18:00–18:30
input bool W37 = false;  // 18:30–19:00
input bool W38 = false;  // 19:00–19:30
input bool W39 = false;  // 19:30–20:00
input bool W40 = false;  // 20:00–20:30
input bool W41 = false;  // 20:30–21:00
input bool W42 = false;  // 21:00–21:30
input bool W43 = false;  // 21:30–22:00
input bool W44 = false;  // 22:00–22:30
input bool W45 = false;  // 22:30–23:00
input bool W46 = false;  // 23:00–23:30
input bool W47 = false;  // 23:30–00:00


// ---- MAE / MFE (FLOATING PNL BASED) ----
double g_maeMoney   = 0.0;   // most negative floating PnL
double g_mfeMoney   = 0.0;   // most positive floating PnL
bool   g_tracking   = false;
ulong  g_ticket     = 0;

datetime g_entryTime = 0;
string   g_csvName   = "trade_stats.csv";

//---------------- Helpers ----------------//
bool IsFlattenTimeDur(datetime barOpen)
{
   MqlDateTime dt; TimeToStruct(barOpen, dt);
   return (dt.hour == FlattenHourDur && dt.min == FlattenMinuteDur);
}

bool IsFlattenTimeEnd(datetime barOpen)
{
   MqlDateTime dt; TimeToStruct(barOpen, dt);
   return (dt.hour == FlattenHourEnd && dt.min == FlattenMinuteEnd);
}

bool g_windows[48];

void InitTradeWindows()
{
   bool tmp[48] = {
      W00,W01,W02,W03,W04,W05,W06,W07,
      W08,W09,W10,W11,W12,W13,W14,W15,
      W16,W17,W18,W19,W20,W21,W22,W23,
      W24,W25,W26,W27,W28,W29,W30,W31,
      W32,W33,W34,W35,W36,W37,W38,W39,
      W40,W41,W42,W43,W44,W45,W46,W47
   };
   ArrayCopy(g_windows, tmp);
}

bool IsTradeWindow(datetime barOpen)
{
   if(!UseTradeWindow) return true;

   MqlDateTime dt;
   TimeToStruct(barOpen, dt);

   int slot = (dt.hour * 60 + dt.min) / 30;
   if(slot < 0 || slot > 47) return false;

   return g_windows[slot];
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
      TimeToString(entryTime, TIME_DATE|TIME_SECONDS),
      TimeToString(exitTime,  TIME_DATE|TIME_SECONDS),
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
int OnInit()
{
   InitTradeWindows();
   return INIT_SUCCEEDED;
}


void OnTick()
{
   datetime barOpen = iTime(_Symbol, _Period, 0);

   // ---- TRACK FLOATING MAE / MFE (tick-based, broker-safe) ----
   if(PositionSelect(_Symbol))
	{
	   double floating = PositionGetDouble(POSITION_PROFIT);

	   if(!g_tracking)
	   {
		  g_tracking  = true;
		  g_ticket    = PositionGetInteger(POSITION_TICKET);
		  g_entryTime = (datetime)PositionGetInteger(POSITION_TIME);
		  g_maeMoney  = floating;
		  g_mfeMoney  = floating;

		  Print("Tracking started ticket=", g_ticket);
	   }

	   // 🔹 update excursions
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
	   for(int i = HistoryDealsTotal() - 1; i >= 0; i--)
	   {
		  ulong deal = HistoryDealGetTicket(i);

		  if(HistoryDealGetInteger(deal, DEAL_POSITION_ID) != g_ticket)
			 continue;

		  double profit = HistoryDealGetDouble(deal, DEAL_PROFIT);
		  realized += profit;

		  if(HistoryDealGetInteger(deal, DEAL_ENTRY) == DEAL_ENTRY_OUT)
		  {
			 exitTime = (datetime)HistoryDealGetInteger(deal, DEAL_TIME);
			 break; // ← exit deal found, stop
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


	// 🔹 flatten during session
	if(UseFlattenDur && IsFlattenTimeDur(barOpen))
	{
	   Print("Flatten cutoff reached DURING SESSION → closing everything");
	   CloseAllPositions();
	   CancelAllOrders();
	   return;
	}

	// 🔹 flatten end of session
	if(UseFlattenEnd && IsFlattenTimeEnd(barOpen))
	{
	   Print("🌙 Flatten cutoff reached → closing everything");
	   CloseAllPositions();
	   CancelAllOrders();
	   return;
	}

	// 🔹 manage trade
	if(PositionsTotal() > 0)
	{
	   ManageOpenPosition();
	   return;
	}

	// 🔹 trade window (ENTRY ONLY)  ✅ MISSING
	if(!IsTradeWindow(barOpen))
	{
	   Print("⏱ Outside trading window → no new entries");
	   CancelOldBuyStops();
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