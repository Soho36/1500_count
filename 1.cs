#region Using declarations
using System;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using System.ComponentModel.DataAnnotations;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class RTLongOnlyTimeWindows : Strategy
    {
        private Order longOrder;
        private double pendingStopPrice;
        private double entryPrice;
        private double riskPerTrade;
        private DateTime lastFlattenDate = Core.Globals.MinDate;
        private bool lastWindowState = false;
        private MarketPosition lastPosition = MarketPosition.Flat;
        // No need for stopLossSubmitted flag anymore!


        // ===== TIME WINDOW INPUTS =====
		[NinjaScriptProperty]
		[Display(Name = "Use Trade Window", Order = 0, GroupName = "Trade Windows")]
		public bool UseTradeWindow { get; set; }

        [NinjaScriptProperty]
        [Display(Name="00:00–00:30", Order=1, GroupName="Trade Windows")]
        public bool W00 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="00:30–01:00", Order=2, GroupName="Trade Windows")]
		public bool W01 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="01:00–01:30", Order=3, GroupName="Trade Windows")]
		public bool W02 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="01:30–02:00", Order=4, GroupName="Trade Windows")]
		public bool W03 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="02:00–02:30", Order=5, GroupName="Trade Windows")]
		public bool W04 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="02:30–03:00", Order=6, GroupName="Trade Windows")]
		public bool W05 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="03:00–03:30", Order=7, GroupName="Trade Windows")]
		public bool W06 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="03:30–04:00", Order=8, GroupName="Trade Windows")]
		public bool W07 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="04:00–04:30", Order=9, GroupName="Trade Windows")]
		public bool W08 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="04:30–05:00", Order=10, GroupName="Trade Windows")]
		public bool W09 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="05:00–05:30", Order=11, GroupName="Trade Windows")]
		public bool W10 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="05:30–06:00", Order=12, GroupName="Trade Windows")]
		public bool W11 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="06:00–06:30", Order=13, GroupName="Trade Windows")]
		public bool W12 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="06:30–07:00", Order=14, GroupName="Trade Windows")]
		public bool W13 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="07:00–07:30", Order=15, GroupName="Trade Windows")]
		public bool W14 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="07:30–08:00", Order=16, GroupName="Trade Windows")]
		public bool W15 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="08:00–08:30", Order=17, GroupName="Trade Windows")]
		public bool W16 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="08:30–09:00", Order=18, GroupName="Trade Windows")]
		public bool W17 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="09:00–09:30", Order=19, GroupName="Trade Windows")]
		public bool W18 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="09:30–10:00", Order=20, GroupName="Trade Windows")]
		public bool W19 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="10:00–10:30", Order=21, GroupName="Trade Windows")]
		public bool W20 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="10:30–11:00", Order=22, GroupName="Trade Windows")]
		public bool W21 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="11:00–11:30", Order=23, GroupName="Trade Windows")]
		public bool W22 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="11:30–12:00", Order=24, GroupName="Trade Windows")]
		public bool W23 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="12:00–12:30", Order=25, GroupName="Trade Windows")]
		public bool W24 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="12:30–13:00", Order=26, GroupName="Trade Windows")]
		public bool W25 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="13:00–13:30", Order=27, GroupName="Trade Windows")]
		public bool W26 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="13:30–14:00", Order=28, GroupName="Trade Windows")]
		public bool W27 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="14:00–14:30", Order=29, GroupName="Trade Windows")]
		public bool W28 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="14:30–15:00", Order=30, GroupName="Trade Windows")]
		public bool W29 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="15:00–15:30", Order=31, GroupName="Trade Windows")]
		public bool W30 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="15:30–16:00", Order=32, GroupName="Trade Windows")]
		public bool W31 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="16:00–16:30", Order=33, GroupName="Trade Windows")]
		public bool W32 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="16:30–17:00", Order=34, GroupName="Trade Windows")]
		public bool W33 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="17:00–17:30", Order=35, GroupName="Trade Windows")]
		public bool W34 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="17:30–18:00", Order=36, GroupName="Trade Windows")]
		public bool W35 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="18:00–18:30", Order=37, GroupName="Trade Windows")]
		public bool W36 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="18:30–19:00", Order=38, GroupName="Trade Windows")]
		public bool W37 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="19:00–19:30", Order=39, GroupName="Trade Windows")]
		public bool W38 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="19:30–20:00", Order=40, GroupName="Trade Windows")]
		public bool W39 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="20:00–20:30", Order=41, GroupName="Trade Windows")]
		public bool W40 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="20:30–21:00", Order=42, GroupName="Trade Windows")]
		public bool W41 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="21:00–21:30", Order=43, GroupName="Trade Windows")]
		public bool W42 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="21:30–22:00", Order=44, GroupName="Trade Windows")]
		public bool W43 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="22:00–22:30", Order=45, GroupName="Trade Windows")]
		public bool W44 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="22:30–23:00", Order=46, GroupName="Trade Windows")]
		public bool W45 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="23:00–23:30", Order=47, GroupName="Trade Windows")]
		public bool W46 { get; set; }

		[NinjaScriptProperty]
		[Display(Name="23:30–00:00", Order=48, GroupName="Trade Windows")]
		public bool W47 { get; set; }

        private bool[] tradeWindows;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name = "RTLongOnlyTimeWindows";
                Calculate = Calculate.OnBarClose;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.UniqueEntries;
                BarsRequiredToTrade = 5;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds = 30;
                StartBehavior = StartBehavior.ImmediatelySubmit;
                IsUnmanaged = false;
                RealtimeErrorHandling = RealtimeErrorHandling.IgnoreAllErrors;

                UseTradeWindow = true;
            }
            else if (State == State.DataLoaded)
            {
                tradeWindows = new bool[]
                {
                    W00,W01,W02,W03,W04,W05,W06,W07,W08,W09,W10,W11,
                    W12,W13,W14,W15,W16,W17,W18,W19,W20,W21,W22,W23,
                    W24,W25,W26,W27,W28,W29,W30,W31,W32,W33,W34,W35,
                    W36,W37,W38,W39,W40,W41,W42,W43,W44,W45,W46,W47
                };
            }
            else if (State == State.Realtime)
            {
                longOrder = null;
                pendingStopPrice = 0;
                entryPrice = 0;
                riskPerTrade = 0;
                Print("=== Strategy entering REALTIME mode ===");
            }
        }

        private bool IsTradeWindow(DateTime time)
        {
            if (!UseTradeWindow)
                return true;

            int minutes = time.Hour * 60 + time.Minute;
            int slot = minutes / 30;

            if (slot < 0 || slot > 47)
                return false;

            return tradeWindows[slot];
        }

        protected override void OnBarUpdate()
        {
			if (State != State.Realtime)
			    return;

            // === End of session FLATTEN LOGIC ===
            if (ToTime(Time[0]) >= 235700 && ToTime(Time[0]) < 235800)
            {
                if (lastFlattenDate.Date != Time[0].Date)
                {
                    lastFlattenDate = Time[0];
					Print($"[{Time[0]}] ❌ End of session FLATTEN → all positions & orders cleared");

                    if (Position.MarketPosition == MarketPosition.Long)
                        ExitLong("DailyFlatten", "Long1");

                    if (longOrder != null &&
                        (longOrder.OrderState == OrderState.Working ||
						 longOrder.OrderState == OrderState.Accepted))
                        CancelOrder(longOrder);
                }
                return;
            }

            if (CurrentBar < BarsRequiredToTrade) return;
            if (State != State.Realtime) return;

            // 🔹 R:R flatten - THIS STAYS HERE because it's a CONDITION, not an order submission
            if (Position.MarketPosition == MarketPosition.Long)
            {
                if (Close[0] - entryPrice >= riskPerTrade)
                {
                    Print($"[{Time[0]}] 🎯 1R reached → reward={Close[0] - entryPrice}, risk={riskPerTrade}");
                    ExitLong("RR_Flatten", "Long1");
                }
                return;
            }

            if (Position.MarketPosition != MarketPosition.Flat)
                return;

            // 🔹 TRADE WINDOW (ENTRY ONLY)
            bool inWindow = IsTradeWindow(Time[0]);

            if (inWindow != lastWindowState)
            {
                Print($"[{Time[0]}] 🪟 Trade window state changed → {(inWindow ? "INSIDE" : "OUTSIDE")}");
                lastWindowState = inWindow;
            }

            if (!inWindow)
            {
                if (longOrder != null &&
                    (longOrder.OrderState == OrderState.Working || longOrder.OrderState == OrderState.Accepted))
                {
                    Print($"[{Time[0]}] ⏱ Outside window → cancelling pending order @ {longOrder.StopPrice}");
                    CancelOrder(longOrder);
                }
                return;
            }

            // 🔹 Red candle logic - ONLY handles entry now
            if (Close[0] < Open[0])
            {
                entryPrice = High[0];
                pendingStopPrice = Low[0];
                riskPerTrade = entryPrice - pendingStopPrice;

                Print($"[{Time[0]}] 🔴 Red candle detected → evaluating entry");
                Print($"[{Time[0]}] ▶ Entry={entryPrice} SL={pendingStopPrice} Risk={riskPerTrade}");

                // REMOVED: SetStopLoss and stopLossSubmitted logic
                // Exit orders will be handled in OnExecutionUpdate

                if (Close[0] > entryPrice)
                {
                    longOrder = EnterLongLimit(0, true, 1, entryPrice, "Long1");
                    Print($"[{Time[0]}] 📥 Submitted BUY LIMIT @ {entryPrice}");
                }
                else
                {
                    longOrder = EnterLongStopLimit(0, true, 1, entryPrice, entryPrice, "Long1");
                    Print($"[{Time[0]}] 📥 Submitted BUY STOP-LIMIT @ {entryPrice}");
                }
            }
        }

        /// <summary>
        /// NEW: OnExecutionUpdate - called IMMEDIATELY when an order fills
        /// This is the FASTEST way to submit protective orders [citation:2]
        /// </summary>
        protected override void OnExecutionUpdate(Execution execution, string executionId,
            double price, int quantity, MarketPosition marketPosition, string orderId, DateTime time)
        {
            // Only process fills for our entry orders
            if (execution.Order != null && execution.Order.Name == "Long1")
            {
                // Check if this is a fill (not a partial fill during order state changes)
                if (execution.Order.OrderState == OrderState.Filled ||
                    (execution.Order.OrderState == OrderState.PartFilled && quantity > 0))
                {
                    // Update entry price with actual fill price
                    entryPrice = execution.Order.AverageFillPrice;

                    // Ensure we have a valid stop price
                    if (pendingStopPrice > 0)
                    {
                        // Calculate risk based on actual fill
                        riskPerTrade = entryPrice - pendingStopPrice;

                        // Submit stop-limit IMMEDIATELY upon fill
                        // Using 4-tick buffer for better fill chances in fast markets
                        double limitPrice = pendingStopPrice - (4 * TickSize);

                        Print($"[{time}] 🚀 Entry FILLED at {entryPrice} - Submitting STOP-LIMIT immediately");
                        Print($"[{time}]    Stop={pendingStopPrice}, Limit={limitPrice}, Risk={riskPerTrade}");

                        // This submits NOW, not waiting for bar close!
                        ExitLongStopLimit(0, true, quantity, pendingStopPrice, limitPrice, "StopLimit", "Long1");
                    }
                }
            }

            // Reset entry order tracking when fully filled
            if (execution.Order != null && execution.Order.Name == "Long1" &&
                execution.Order.OrderState == OrderState.Filled)
            {
                longOrder = null;
            }
        }

        /// <summary>
        /// Optional: Track order updates for debugging
        /// </summary>
        protected override void OnOrderUpdate(Order order, double limitPrice, double stopPrice,
            int quantity, int filled, double averageFillPrice, OrderState orderState,
            DateTime time, ErrorCode error, string nativeError)
        {
            // Track our entry order
            if (order.Name == "Long1")
            {
                longOrder = order;
            }

            // Optional: Print order updates for debugging
            // Print($"[{time}] Order Update: {order.Name} - {orderState}");
        }
    }
}