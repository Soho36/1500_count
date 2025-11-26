#region Using declarations
using System;
using System.IO;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class NT8LongOnlyAccRotate : Strategy
    {
        // -------------------------------------------------------
        // 🔷 ROTATION SYSTEM (only thing you change per script)
        // -------------------------------------------------------
        private int InstanceIndex = 1;       // <── Hardcode: 1, 2, or 3
        private string instanceFile;
        private int activeInstance = -1;
        // -------------------------------------------------------

        private Order longOrder;
        private double pendingStopPrice;
        private double entryPrice;
        private double riskPerTrade;
		private DateTime lastFlattenDate = Core.Globals.MinDate;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name = "NT8LongOnlyAccRotate";
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
            else if (State == State.Configure)
            {
                // Shared file to sync all instances
                instanceFile = Path.Combine(
                    NinjaTrader.Core.Globals.UserDataDir,
                    "bin", "Custom", "ActiveInstance.txt"
                );
            }
            else if (State == State.DataLoaded)
            {
                // Create file if missing
                if (!File.Exists(instanceFile))
                {
                    File.WriteAllText(instanceFile, "1");
                    Print("Created ActiveInstance.txt → initialized to 1");
                }
            }
            else if (State == State.Realtime)
            {
                longOrder = null;
                pendingStopPrice = 0;
                entryPrice = 0;
                riskPerTrade = 0;
                Print($"=== Instance #{InstanceIndex} entering REALTIME ===");
            }
        }

        protected override void OnBarUpdate()
        {
            // -------------------------------------------------------
            // 🔷 LOAD ACTIVE INSTANCE FROM FILE
            // -------------------------------------------------------
            try
            {
                string txt = File.ReadAllText(instanceFile).Trim();
                if (!int.TryParse(txt, out activeInstance))
                    activeInstance = 1;
            }
            catch (Exception ex)
            {
                Print("Error reading instance file: " + ex.Message);
                return;
            }

            // -------------------------------------------------------
            // 🔷 IF THIS IS NOT THE ACTIVE INSTANCE → WAIT
            // -------------------------------------------------------
            if (activeInstance != InstanceIndex)
            {
                Print($"[{Time[0]}] Instance #{InstanceIndex} waiting… Active = #{activeInstance}");
                return;
            }

            // -------------------------------------------------------
            // 🔷 THIS INSTANCE IS ACTIVE → RUN TRADING LOGIC
            // -------------------------------------------------------
            // DAILY FLATTEN 17:00
            if (ToTime(Time[0]) >= 170000 && ToTime(Time[0]) < 170100)
            {
                if (lastFlattenDate.Date != Time[0].Date)
                {
                    lastFlattenDate = Time[0];

                    if (Position.MarketPosition == MarketPosition.Long)
                    {
                        Print($"[{Time[0]}] 17:00 flatten → closing long");
                        ExitLong("DailyFlatten", "Long1");
                    }

                    if (longOrder != null &&
                        (longOrder.OrderState == OrderState.Working ||
                         longOrder.OrderState == OrderState.Accepted))
                    {
                        Print($"[{Time[0]}] 17:00 flatten → cancel pending entry {longOrder.StopPrice}");
                        CancelOrder(longOrder);
                    }
                }
                return;
            }

            if (CurrentBar < BarsRequiredToTrade) return;
            if (State != State.Realtime) return;

            Print($"[{Time[0]}] ACTIVE Instance #{InstanceIndex} | Pos={Position.MarketPosition}");

            // -------------------------------------------------------
            // 1:1 R/R EXIT
            // -------------------------------------------------------
            if (Position.MarketPosition == MarketPosition.Long)
            {
                double reward = Close[0] - entryPrice;
                if (reward >= riskPerTrade)
                {
                    Print($"[{Time[0]}] [RR EXIT] 1:1 R/R reached → exit long");
                    ExitLong("RR_Flatten", "Long1");

                    RotateToNext();     // <-- ROTATE AFTER TRADE IS DONE
                }
                return;
            }

            if (Position.MarketPosition != MarketPosition.Flat)
                return;

            // -------------------------------------------------------
            // ENTRY SETUP (Red candle)
            // -------------------------------------------------------
            if (Close[0] < Open[0])
            {
                entryPrice = High[0] + TickSize;
                pendingStopPrice = Low[0] - TickSize;
                riskPerTrade = entryPrice - pendingStopPrice;

                SetStopLoss("Long1", CalculationMode.Price, pendingStopPrice, false);

                double stopPrice = entryPrice;

                if (Close[0] > stopPrice)
                {
                    longOrder = EnterLongLimit(0, true, 1, stopPrice, "Long1");
                    Print($"[{Time[0]}] Limit entry @ {stopPrice} (slippage fallback)");
                }
                else
                {
                    longOrder = EnterLongStopLimit(0, true, 1, stopPrice, stopPrice, "Long1");
                    Print($"[{Time[0]}] Submitted Buy StopLimit @ {stopPrice}");
                }
            }
        }

        // -------------------------------------------------------
        // 🔷 ROTATE CONTROL TO NEXT INSTANCE (1 → 2 → 3 → 1)
        // -------------------------------------------------------
        private void RotateToNext()
        {
            int next = InstanceIndex + 1;
            if (next > 3) next = 1;

            try
            {
                File.WriteAllText(instanceFile, next.ToString());
                Print($"Instance #{InstanceIndex} finished → Rotating control to #{next}");
            }
            catch (Exception ex)
            {
                Print("Error writing rotation file: " + ex.Message);
            }
        }

        protected override void OnExecutionUpdate(Cbi.Execution execution, string executionId, double price, int quantity,
            Cbi.MarketPosition marketPosition, string orderId, DateTime time)
        {
            if (execution.Order == null)
                return;

            if (execution.Order.Name == "Long1" &&
                execution.Order.OrderState == OrderState.Filled &&
                marketPosition == MarketPosition.Long)
            {
                Print($"[{time}] Entry filled @ {price}, SL already set @ {pendingStopPrice}");
            }
        }
    }
}
