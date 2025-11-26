#region Using declarations
using System;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class NT8LongOnly : Strategy
    {
        private Order longOrder;
        private double pendingStopPrice;
        private double entryPrice;
        private double riskPerTrade;
		private DateTime lastFlattenDate = Core.Globals.MinDate;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name = "NT8LongOnly";
                Calculate = Calculate.OnBarClose;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.UniqueEntries;
                BarsRequiredToTrade = 5;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds = 30;
                StartBehavior = StartBehavior.ImmediatelySubmit;
                IsUnmanaged = false;   // ✅ managed mode
                RealtimeErrorHandling = RealtimeErrorHandling.IgnoreAllErrors;
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

        protected override void OnBarUpdate()
		{
			// === DAILY FLATTEN LOGIC ===
			if (ToTime(Time[0]) >= 170000 && ToTime(Time[0]) < 170100) // between 17:00:00 and 17:01:00
			{
				if (lastFlattenDate.Date != Time[0].Date)
				{
					lastFlattenDate = Time[0];

					// 1️⃣ Close any open long position
					if (Position.MarketPosition == MarketPosition.Long)
					{
						Print($"[{Time[0]}] ⏰ 17:00 flatten trigger → closing open long position");
						ExitLong("DailyFlatten", "Long1");
					}

					// 2️⃣ Cancel any working buy stop/limit order
					if (longOrder != null &&
						(longOrder.OrderState == OrderState.Working || longOrder.OrderState == OrderState.Accepted))
					{
						Print($"[{Time[0]}] ⏰ 17:00 flatten trigger → cancelling pending entry order @ {longOrder.StopPrice}");
						CancelOrder(longOrder);
					}
				}

				return;	// skip the rest of the logic this bar
			}

			if (CurrentBar < BarsRequiredToTrade) return;
			if (State != State.Realtime) return;

			// Debug info
			Print($"[{Time[0]}] OnBarUpdate | H={High[0]} L={Low[0]} Pos={Position.MarketPosition}");

			// 🔹 Flatten if 1:1 R/R reached
			if (Position.MarketPosition == MarketPosition.Long)
			{
				double reward = Close[0] - entryPrice;
				if (reward >= riskPerTrade)
				{
					Print($"[{Time[0]}] [FLATTEN] 1:1 R/R reached (reward={reward}, risk={riskPerTrade}) → closing position");
					ExitLong("RR_Flatten", "Long1");
				}
				return; // don’t place new orders while in position
			}

			// Skip if not flat
			if (Position.MarketPosition != MarketPosition.Flat)
				return;

			// Only act on red candles (Close < Open)
			if (Close[0] < Open[0])
			{
				entryPrice = High[0] + TickSize;     // intended stop entry price
				pendingStopPrice = Low[0] - TickSize; // stop loss below candle low
				riskPerTrade = entryPrice - pendingStopPrice;

				// ✅ Attach SL BEFORE entry
				SetStopLoss("Long1", CalculationMode.Price, pendingStopPrice, false);

				// ⚙️ Define stop/limit price
				double stopPrice = entryPrice;

				// ✅ Pre-check for slippage
				if (Close[0] > stopPrice)
				{
					// ⚠️ Price already above intended stop → fallback to limit order
					longOrder = EnterLongLimit(0, true, 1, stopPrice, "Long1");
					Print($"[{Time[0]}] ⚠️ Price jumped above stop ({Close[0]} > {stopPrice}) → fallback to Buy Limit @ {stopPrice}");
				}
				else
				{
					// ✅ Normal case → Buy Stop Limit
					longOrder = EnterLongStopLimit(0, true, 1, stopPrice, stopPrice, "Long1");
					Print($"[{Time[0]}] >>> Submitted Buy Stop Limit @ {stopPrice}, SL will be @ {pendingStopPrice}");
				}
			}
		}


        protected override void OnExecutionUpdate(Cbi.Execution execution, string executionId, double price, int quantity,
            Cbi.MarketPosition marketPosition, string orderId, DateTime time)
        {
            if (execution.Order == null)
                return;

            // Just debug info now — no SetStopLoss here
            if (execution.Order.Name == "Long1" &&
                execution.Order.OrderState == OrderState.Filled &&
                marketPosition == MarketPosition.Long)
            {
                Print($"[{time}] [ENTRY FILLED] Long entry filled @ {price}, SL is already set @ {pendingStopPrice}");
            }

            if (Position.MarketPosition == MarketPosition.Flat)
            {
                Print($"[{time}] Flat → no active SL");
            }
        }
    }
}