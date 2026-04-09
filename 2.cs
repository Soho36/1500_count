//+------------------------------------------------------------------+
//| Red-Green Breakout EA: dynamic exit + time + month + weekday filters |
//+------------------------------------------------------------------+
#property strict

input double Lots           = 1.0;
input int FixedTPPoints 	= 5;  // Take profit distance in whole points (e.g. 5 = 5 index points)
input int    Slippage       = 5;

// ======== CANDLE RANGE FILTER ========
input bool   UseCandleRangeFilter = false;
input double MaxCandleRange       = 50.0;
input double MinCandleRange       = 5.0;

// ======== WEEKDAY FILTERING OPTIONS ========
enum WEEKDAY_FILTER_MODE
{
   WEEKDAY_DISABLED,
   WEEKDAY_INDIVIDUAL,
   WEEKDAY_GROUPED
};

input WEEKDAY_FILTER_MODE WeekdayFilterMode = WEEKDAY_DISABLED;

input bool TradeMonday     = true;
input bool TradeTuesday    = true;
input bool TradeWednesday  = true;
input bool TradeThursday   = true;
input bool TradeFriday     = true;
input bool TradeSaturday   = true;
input bool TradeSunday     = true;

input bool TradeWeekdays   = true;
input bool TradeWeekend    = false;

// ======== MONTH FILTERING OPTIONS ========
enum MONTH_FILTER_MODE
{
   MONTH_DISABLED,
   MONTH_INDIVIDUAL,
   MONTH_GROUPED
};

input MONTH_FILTER_MODE MonthFilterMode = MONTH_DISABLED;

input bool TradeJanuary    = true;
input bool TradeFebruary   = true;
input bool TradeMarch      = true;
input bool TradeApril      = true;
input bool TradeMay        = true;
input bool TradeJune       = true;
input bool TradeJuly       = true;
input bool TradeAugust     = true;
input bool TradeSeptember  = true;
input bool TradeOctober    = true;
input bool TradeNovember   = true;
input bool TradeDecember   = true;

input bool TradeWinter     = true;
input bool TradeSpring     = true;
input bool TradeSummer     = true;
input bool TradeAutumn     = true;

// ======== SESSION FILTERING ========
input bool   UseFlattenDur     = true;
input int    FlattenHourDur    = 14;
input int    FlattenMinuteDur  = 00;

input bool   UseFlattenEnd     = true;
input int    FlattenHourEnd    = 20;
input int    FlattenMinuteEnd  = 00;

// ======== TIME WINDOW FILTERING ========
input bool   UseTradeWindow   = true;

input bool W0000W0100 = false;

input bool W0100W0130 = false;
input bool W0130W0200 = false;

input bool W0200W0300 = false;
input bool W0300W0400 = false;
input bool W0400W0500 = false;
input bool W0500W0600 = false;
input bool W0600W0700 = false;
input bool W0700W0800 = false;
input bool W0800W0900 = false;
input bool W0900W1000 = false;

input bool W1000W1100 = false;
input bool W1100W1200 = false;
input bool W1200W1300 = false;
input bool W1300W1400 = false;
input bool W1400W1500 = false;
input bool W1500W1600 = false;
input bool W1600W1700 = false;
input bool W1700W1800 = false;
input bool W1800W1900 = false;
input bool W1900W2000 = false;
input bool W2000W2100 = false;
input bool W2100W2200 = false;
input bool W2200W2300 = false;

input bool W2300W2330 = false;
input bool W2330W0000 = false;

bool windows[26] =
{
   W0000W0100,
   W0100W0130, W0130W0200,
   W0200W0300, W0300W0400, W0400W0500, W0500W0600,
   W0600W0700, W0700W0800, W0800W0900, W0900W1000,
   W1000W1100, W1100W1200, W1200W1300, W1300W1400,
   W1400W1500, W1500W1600, W1600W1700, W1700W1800,
   W1800W1900, W1900W2000, W2000W2100, W2100W2200,
   W2200W2300,
   W2300W2330, W2330W0000
};

string GetSessionName(int slot)
{
   if(slot == 0) return "MARKET CLOSED";
   if(slot >= 1 && slot <= 10) return "MORNING";
   if(slot >= 11 && slot <= 23) return "MAIN";
   if(slot >= 24 && slot <= 25) return "EVENING";
   return "UNKNOWN";
}

// ======== MAE / MFE (FLOATING PNL BASED) ========
double g_maeMoney   = 0.0;
double g_mfeMoney   = 0.0;
bool   g_tracking   = false;
ulong  g_ticket     = 0;

datetime g_entryTime = 0;
string   g_csvName   = "trade_stats.csv";

// ======== HELPER FUNCTIONS ========
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

bool IsTradeWindow(datetime barOpen)
{
   if(!UseTradeWindow)
      return true;

   MqlDateTime dt;
   TimeToStruct(barOpen, dt);

   int totalMinutes = dt.hour * 60 + dt.min;
   int slot;

   if(totalMinutes < 60)
      slot = 0;
   else if(totalMinutes >= 60 && totalMinutes < 600)
   {
      if(totalMinutes < 120)
         slot = 1 + ((totalMinutes - 60) / 30);
      else
         slot = 3 + (dt.hour - 2);
   }
   else if(totalMinutes >= 600 && totalMinutes < 1380)
      slot = 11 + (dt.hour - 10);
   else
      slot = 24 + ((totalMinutes - 1380) / 30);

   return windows[slot];
}

void DisplayTradeWindowStatus(datetime barOpen)
{
   if(!UseTradeWindow) return;

   MqlDateTime dt;
   TimeToStruct(barOpen, dt);

   int totalMinutes = dt.hour * 60 + dt.min;
   string borderLine = "";

   if(totalMinutes < 60)
      borderLine = "═══════════════════════════════════════════════";
   else if(totalMinutes == 60)
      borderLine = "═══════════════ MORNING SESSION START ═══════════════";
   else if(totalMinutes == 600)
      borderLine = "════════════════ MAIN SESSION START ════════════════";
   else if(totalMinutes == 1380)
      borderLine = "═══════════════ EVENING SESSION START ═══════════════";

   if(borderLine != "")
      Print(borderLine);
}

bool IsCandleInRange(double high, double low)
{
   if(!UseCandleRangeFilter)
      return true;

   double rangePoints = (high - low) / _Point;

   if(rangePoints > MaxCandleRange)
   {
      Print("⚠️ Candle range filter: Range ", DoubleToString(rangePoints, 2), " points > Max ", DoubleToString(MaxCandleRange, 2), " points - Skipping");
      return false;
   }
   if(rangePoints < MinCandleRange)
   {
      Print("⚠️ Candle range filter: Range ", DoubleToString(rangePoints, 2), " points < Min ", DoubleToString(MinCandleRange, 2), " points - Skipping");
      return false;
   }
   return true;
}

bool IsWeekdayAllowed(datetime barOpen)
{
   if(WeekdayFilterMode == WEEKDAY_DISABLED)
      return true;

   MqlDateTime dt;
   TimeToStruct(barOpen, dt);
   int weekday = dt.day_of_week;

   if(WeekdayFilterMode == WEEKDAY_INDIVIDUAL)
   {
      switch(weekday)
      {
         case 1:  return TradeMonday;
         case 2:  return TradeTuesday;
         case 3:  return TradeWednesday;
         case 4:  return TradeThursday;
         case 5:  return TradeFriday;
         case 6:  return TradeSaturday;
         case 0:  return TradeSunday;
      }
   }
   else if(WeekdayFilterMode == WEEKDAY_GROUPED)
   {
      if(weekday >= 1 && weekday <= 5) return TradeWeekdays;
      else if(weekday == 6 || weekday == 0) return TradeWeekend;
   }
   return false;
}

string GetWeekdayName(int weekday)
{
   switch(weekday)
   {
      case 1:  return "Monday";
      case 2:  return "Tuesday";
      case 3:  return "Wednesday";
      case 4:  return "Thursday";
      case 5:  return "Friday";
      case 6:  return "Saturday";
      case 0:  return "Sunday";
   }
   return "Unknown";
}

bool IsMonthAllowed(datetime barOpen)
{
   if(MonthFilterMode == MONTH_DISABLED)
      return true;

   MqlDateTime dt;
   TimeToStruct(barOpen, dt);
   int month = dt.mon;

   if(MonthFilterMode == MONTH_INDIVIDUAL)
   {
      switch(month)
      {
         case 1:  return TradeJanuary;
         case 2:  return TradeFebruary;
         case 3:  return TradeMarch;
         case 4:  return TradeApril;
         case 5:  return TradeMay;
         case 6:  return TradeJune;
         case 7:  return TradeJuly;
         case 8:  return TradeAugust;
         case 9:  return TradeSeptember;
         case 10: return TradeOctober;
         case 11: return TradeNovember;
         case 12: return TradeDecember;
      }
   }
   else if(MonthFilterMode == MONTH_GROUPED)
   {
      if(month == 12 || month == 1 || month == 2) return TradeWinter;
      else if(month >= 3 && month <= 5)            return TradeSpring;
      else if(month >= 6 && month <= 8)            return TradeSummer;
      else if(month >= 9 && month <= 11)           return TradeAutumn;
   }
   return false;
}

string GetMonthName(int month)
{
   switch(month)
   {
      case 1:  return "January";
      case 2:  return "February";
      case 3:  return "March";
      case 4:  return "April";
      case 5:  return "May";
      case 6:  return "June";
      case 7:  return "July";
      case 8:  return "August";
      case 9:  return "September";
      case 10: return "October";
      case 11: return "November";
      case 12: return "December";
   }
   return "Unknown";
}

// ======== TP PRICE CALCULATION ========
double CalcTPPrice(double entryPrice)
{
   return entryPrice + FixedTPPoints;
}

// ======== CSV FUNCTIONS ========
void SaveTradeStats(double realized, datetime entryTime, datetime exitTime)
{
   int f = FileOpen(g_csvName, FILE_READ|FILE_WRITE|FILE_CSV|FILE_SHARE_WRITE);
   if(f == INVALID_HANDLE)
   {
      Print("File open failed ", GetLastError());
      return;
   }

   if(FileSize(f) == 0)
      FileWrite(f, "ticket", "entry_time", "exit_time", "mae_money", "mfe_money", "trade_profit");

   FileSeek(f, 0, SEEK_END);
   FileWrite(
      f,
      (long)g_ticket,
      TimeToString(entryTime, TIME_DATE|TIME_SECONDS),
      TimeToString(exitTime, TIME_DATE|TIME_SECONDS),
      g_maeMoney,
      g_mfeMoney,
      realized
   );

   FileClose(f);
}

// ======== TRADE MANAGEMENT FUNCTIONS ========
void CancelAllOrders()
{
   for(int i = OrdersTotal() - 1; i >= 0; --i)
   {
      ulong ticket = OrderGetTicket(i);
      if(ticket == 0) continue;
      if(!OrderSelect(ticket)) continue;

      MqlTradeRequest req = {};
      MqlTradeResult  res = {};
      req.action = TRADE_ACTION_REMOVE;
      req.order  = ticket;

      if(!OrderSend(req, res))
         Print("❌ CancelAllOrders failed ticket=", ticket, " err=", GetLastError());
      else
         Print("✅ Cancelled order ticket=", ticket);
   }
}

void CloseAllPositions()
{
   for(int i = PositionsTotal() - 1; i >= 0; --i)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(!PositionSelectByTicket(ticket)) continue;

      string sym = PositionGetString(POSITION_SYMBOL);
      double vol = PositionGetDouble(POSITION_VOLUME);
      long   typ = PositionGetInteger(POSITION_TYPE);

      MqlTradeRequest req = {};
      MqlTradeResult  res = {};
      req.action    = TRADE_ACTION_DEAL;
      req.symbol    = sym;
      req.volume    = vol;
      req.deviation = Slippage;

      if(typ == POSITION_TYPE_BUY)
      {
         req.type  = ORDER_TYPE_SELL;
         req.price = SymbolInfoDouble(sym, SYMBOL_BID);
      }
      else
      {
         req.type  = ORDER_TYPE_BUY;
         req.price = SymbolInfoDouble(sym, SYMBOL_ASK);
      }

      if(!OrderSend(req, res))
         Print("❌ CloseAllPositions failed pos#", ticket, " err=", GetLastError());
      else
         Print("✅ Closed position #", ticket);
   }
}

void CancelOldBuyStops()
{
   for(int i = OrdersTotal() - 1; i >= 0; --i)
   {
      ulong ticket = OrderGetTicket(i);
      if(ticket == 0) continue;
      if(!OrderSelect(ticket)) continue;

      int type = (int)OrderGetInteger(ORDER_TYPE);
      if(type != ORDER_TYPE_BUY_STOP) continue;

      MqlTradeRequest req = {};
      MqlTradeResult  res = {};
      req.action = TRADE_ACTION_REMOVE;
      req.order  = ticket;

      if(!OrderSend(req, res))
         Print("❌ Failed to cancel BuyStop ticket=", ticket, " err=", GetLastError());
      else
         Print("✅ Cancelled BuyStop ticket=", ticket);
   }
}

// ======== DISPLAY CURRENT SETTINGS ========
void DisplaySettings()
{
   double tickSize  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);   // 0.25
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);  // 0.50
   Print("╔════════════════════════════════════════════════════════════╗");
   Print("║                    EA SETTINGS                             ║");
   Print("╚════════════════════════════════════════════════════════════╝");
   Print("Lots: ", Lots, ", Fixed TP: ", FixedTPPoints, " points (",
      FixedTPPoints / tickSize, " ticks, $",
      (FixedTPPoints / tickSize) * tickValue * Lots, " per trade)");

   Print("┌────────────────────────────────────────────────────────────┐");
   Print("│ CANDLE RANGE FILTER                                        │");
   Print("├────────────────────────────────────────────────────────────┤");
   Print("│ Status: ", UseCandleRangeFilter ? "ENABLED" : "DISABLED");
   if(UseCandleRangeFilter)
   {
      Print("│ Max Range: ", MaxCandleRange, " points");
      Print("│ Min Range: ", MinCandleRange, " points");
   }
   Print("└────────────────────────────────────────────────────────────┘");

   Print("┌────────────────────────────────────────────────────────────┐");
   Print("│ WEEKDAY FILTERING                                          │");
   Print("├────────────────────────────────────────────────────────────┤");
   if(WeekdayFilterMode == WEEKDAY_DISABLED)
      Print("│ Status: DISABLED");
   else if(WeekdayFilterMode == WEEKDAY_INDIVIDUAL)
   {
      Print("│ Status: INDIVIDUAL WEEKDAYS");
      Print("│ Mon:", TradeMonday, " Tue:", TradeTuesday, " Wed:", TradeWednesday,
            " Thu:", TradeThursday, " Fri:", TradeFriday);
      Print("│ Sat:", TradeSaturday, " Sun:", TradeSunday);
   }
   else if(WeekdayFilterMode == WEEKDAY_GROUPED)
   {
      Print("│ Status: WEEKDAY GROUPS");
      Print("│ Weekdays (Mon-Fri): ", TradeWeekdays);
      Print("│ Weekend (Sat-Sun): ", TradeWeekend);
   }
   Print("└────────────────────────────────────────────────────────────┘");

   Print("┌────────────────────────────────────────────────────────────┐");
   Print("│ MONTH FILTERING                                            │");
   Print("├────────────────────────────────────────────────────────────┤");
   if(MonthFilterMode == MONTH_DISABLED)
      Print("│ Status: DISABLED");
   else if(MonthFilterMode == MONTH_INDIVIDUAL)
   {
      Print("│ Status: INDIVIDUAL MONTHS");
      Print("│ Jan:", TradeJanuary, " Feb:", TradeFebruary, " Mar:", TradeMarch,
            " Apr:", TradeApril, " May:", TradeMay, " Jun:", TradeJune);
      Print("│ Jul:", TradeJuly, " Aug:", TradeAugust, " Sep:", TradeSeptember,
            " Oct:", TradeOctober, " Nov:", TradeNovember, " Dec:", TradeDecember);
   }
   else if(MonthFilterMode == MONTH_GROUPED)
   {
      Print("│ Status: SEASONAL GROUPS");
      Print("│ Winter (Dec-Jan-Feb): ", TradeWinter);
      Print("│ Spring (Mar-Apr-May): ", TradeSpring);
      Print("│ Summer (Jun-Jul-Aug): ", TradeSummer);
      Print("│ Autumn (Sep-Oct-Nov): ", TradeAutumn);
   }
   Print("└────────────────────────────────────────────────────────────┘");

   Print("┌────────────────────────────────────────────────────────────┐");
   Print("│ TIME WINDOW FILTERING - SESSIONS                           │");
   Print("├────────────────────────────────────────────────────────────┤");
   Print("│ Status: ", UseTradeWindow ? "ENABLED" : "DISABLED");
   if(UseTradeWindow)
   {
      Print("├────────────────────────────────────────────────────────────┤");
      Print("│ SESSION 1: MARKET CLOSED (00:00-01:00) - 1 hour           │");
      Print("├────────────────────────────────────────────────────────────┤");
      Print("│ SESSION 2: MORNING (01:00-10:00)                          │");
      Print("│   - 01:00-02:00: 30-min intervals                         │");
      Print("│   - 02:00-10:00: 1-hour intervals                         │");
      Print("├────────────────────────────────────────────────────────────┤");
      Print("│ SESSION 3: MAIN (10:00-23:00) - 1-hour intervals          │");
      Print("├────────────────────────────────────────────────────────────┤");
      Print("│ SESSION 4: EVENING (23:00-00:00) - 30-min intervals       │");
   }
   Print("└────────────────────────────────────────────────────────────┘");

   Print("┌────────────────────────────────────────────────────────────┐");
   Print("│ FLATTEN TIMES                                              │");
   Print("├────────────────────────────────────────────────────────────┤");
   Print("│ During Session: ", UseFlattenDur ? "Yes (" + (string)FlattenHourDur + ":" + (string)FlattenMinuteDur + ")" : "No");
   Print("│ End of Session: ", UseFlattenEnd ? "Yes (" + (string)FlattenHourEnd + ":" + (string)FlattenMinuteEnd + ")" : "No");
   Print("└────────────────────────────────────────────────────────────┘");
}

// ======== EA CORE ========
int OnInit()
{
   if(FileIsExist(g_csvName))
   {
      if(FileDelete(g_csvName))
         Print("Old trade_stats.csv deleted");
      else
         Print("Failed to delete old CSV. Error=", GetLastError());
   }

   DisplaySettings();
   return(INIT_SUCCEEDED);
}

void OnTick()
{
   // ---- TRACK FLOATING MAE / MFE (tick-based) ----
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

      g_maeMoney = MathMin(g_maeMoney, floating);
      g_mfeMoney = MathMax(g_mfeMoney, floating);
   }
   else if(g_tracking)
   {
      double realized = 0.0;
      datetime exitTime = 0;

      if(HistorySelect(g_entryTime - 86400, TimeCurrent()))
      {
         for(int i = HistoryDealsTotal() - 1; i >= 0; i--)
         {
            ulong deal = HistoryDealGetTicket(i);
            if(HistoryDealGetInteger(deal, DEAL_POSITION_ID) != g_ticket) continue;

            double profit = HistoryDealGetDouble(deal, DEAL_PROFIT);
            realized += profit;

            if(HistoryDealGetInteger(deal, DEAL_ENTRY) == DEAL_ENTRY_OUT)
            {
               exitTime = (datetime)HistoryDealGetInteger(deal, DEAL_TIME);
               break;
            }
         }
      }

      g_maeMoney = MathMin(g_maeMoney, realized);
      g_mfeMoney = MathMax(g_mfeMoney, realized);
      SaveTradeStats(realized, g_entryTime, exitTime);
      g_tracking = false;
   }

   datetime barOpen = iTime(_Symbol, _Period, 0);

   // 🔹 Flatten checks — always evaluated first, even with open position
   if(UseFlattenDur && IsFlattenTimeDur(barOpen))
   {
      Print("Flatten cutoff reached DURING SESSION → closing everything");
      CloseAllPositions();
      CancelAllOrders();
      return;
   }
   if(UseFlattenEnd && IsFlattenTimeEnd(barOpen))
   {
      Print("🌙 Flatten cutoff reached → closing everything");
      CloseAllPositions();
      CancelAllOrders();
      return;
   }

   // 🔹 If position is open, TP is handled by the broker — nothing to do here
   if(PositionsTotal() > 0) return;

   // ---- Per-bar logic below ----
   static datetime lastBar = 0;
   if(barOpen == lastBar) return;
   lastBar = barOpen;

   MqlDateTime dt;
   TimeToStruct(barOpen, dt);
   string monthName   = GetMonthName(dt.mon);
   string weekdayName = GetWeekdayName(dt.day_of_week);

   DisplayTradeWindowStatus(barOpen);

   if(WeekdayFilterMode != WEEKDAY_DISABLED && !IsWeekdayAllowed(barOpen))
   {
      Print("📅 Weekday Filter: ", weekdayName, " not allowed for trading");
      CancelOldBuyStops();
      return;
   }

   if(MonthFilterMode != MONTH_DISABLED && !IsMonthAllowed(barOpen))
   {
      Print("📅 Month Filter: ", monthName, " not allowed for trading");
      CancelOldBuyStops();
      return;
   }

   if(!IsTradeWindow(barOpen))
   {
      Print("⏱ Outside trading window → no new entries");
      CancelOldBuyStops();
      return;
   }

   double o1 = iOpen(_Symbol, _Period, 1);
   double h1 = iHigh(_Symbol, _Period, 1);
   double l1 = iLow(_Symbol, _Period, 1);
   double c1 = iClose(_Symbol, _Period, 1);

   if(!IsCandleInRange(h1, l1))
   {
      Print("⚠️ Candle range outside allowed limits - Skipping signal");
      CancelOldBuyStops();
      return;
   }

   // BUY-STOP ORDER AT PREVIOUS RED CANDLE HIGH
   if(c1 < o1)
   {
      Print("🔴 Red candle on ", weekdayName, " in ", monthName, " → refresh BuyStop");
      CancelOldBuyStops();

      double entry = h1;
      double stop  = l1;
      double risk  = entry - stop;
      if(risk <= 0.0) return;

      // Calculate hard TP price from fixed money target
      double tp = CalcTPPrice(entry);

      MqlTradeRequest req = {};
      MqlTradeResult  res = {};
      req.action       = TRADE_ACTION_PENDING;
      req.symbol       = _Symbol;
      req.volume       = Lots;
      req.type         = ORDER_TYPE_BUY_STOP;
      req.price        = entry;
      req.sl           = stop;
      req.tp           = tp;
      req.deviation    = Slippage;
      req.type_filling = ORDER_FILLING_RETURN;

      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      if(ask >= entry)
      {
         Print("⚠️ Gap above entry at session open → skipping BuyStop placement");
         return;
      }

      if(!OrderSend(req, res))
        Print("❌ Place BuyStop fail err=", GetLastError());
      else
		Print("🚀 BuyStop placed @", entry, " SL=", stop, " TP=", tp,
			  " (", FixedTPPoints, " pts, ", weekdayName, ", ", monthName, ")");
   }
}