#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
#endregion

namespace NinjaTrader.NinjaScript.Strategies
{
    public class RedGreenBreakoutEA : Strategy
    {
        #region Trade Parameters
        [NinjaScriptProperty]
        [Range(1, int.MaxValue)]
        [Display(Name = "Quantity (contracts)", Order = 1, GroupName = "Trade Parameters")]
        public int Quantity { get; set; }

        [NinjaScriptProperty]
        [Range(0.01, double.MaxValue)]
        [Display(Name = "Risk/Reward Ratio", Order = 2, GroupName = "Trade Parameters",
                 Description = "TP = signalHigh + candleRange * RR. Anchored to signal candle, not to entry.")]
        public double RiskReward { get; set; }

        [NinjaScriptProperty]
        [Range(0.0, 100.0)]
        [Display(Name = "Limit Offset % (0=High, 100=Low)", Order = 3, GroupName = "Trade Parameters",
                 Description = "0% places the limit at the candle High, 100% at the candle Low. Typical range: 10-90.")]
        public double LimitOffsetPercent { get; set; }
        #endregion

        #region Candle Range Filter
        [NinjaScriptProperty]
        [Display(Name = "Use Candle Range Filter", Order = 1, GroupName = "Candle Range Filter")]
        public bool UseCandleRangeFilter { get; set; }

        [NinjaScriptProperty]
        [Range(0, double.MaxValue)]
        [Display(Name = "Max Candle Range (ticks)", Order = 2, GroupName = "Candle Range Filter")]
        public double MaxCandleRange { get; set; }

        [NinjaScriptProperty]
        [Range(0, double.MaxValue)]
        [Display(Name = "Min Candle Range (ticks)", Order = 3, GroupName = "Candle Range Filter")]
        public double MinCandleRange { get; set; }
        #endregion

        #region Flatten End of Session
        [NinjaScriptProperty]
        [Display(Name = "Use Flatten End", Order = 1, GroupName = "Flatten End of Session")]
        public bool UseFlattenEnd { get; set; }

        [NinjaScriptProperty]
        [Range(0, 23)]
        [Display(Name = "Flatten Hour", Order = 2, GroupName = "Flatten End of Session")]
        public int FlattenHourEnd { get; set; }

        [NinjaScriptProperty]
        [Range(0, 59)]
        [Display(Name = "Flatten Minute", Order = 3, GroupName = "Flatten End of Session")]
        public int FlattenMinuteEnd { get; set; }
        #endregion

        #region Time Window Filter
        [NinjaScriptProperty]
        [Display(Name = "Use Time Trade Window", Order = 1, GroupName = "Time Window Filter")]
        public bool UseTradeWindow { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Allowed Hours (comma-separated, 0-23)", Order = 2, GroupName = "Time Window Filter",
                 Description = "Example: 10,11,12,13,14,15 - new setups allowed only during these hours.")]
        public string AllowedHours { get; set; }
        #endregion

        #region Weekday Filter
        [NinjaScriptProperty]
        [Display(Name = "Use Weekday Filter", Order = 1, GroupName = "Weekday Filter")]
        public bool UseWeekdayFilter { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Trade Monday", Order = 2, GroupName = "Weekday Filter")] public bool TradeMonday { get; set; }
        [NinjaScriptProperty]
        [Display(Name = "Trade Tuesday", Order = 3, GroupName = "Weekday Filter")] public bool TradeTuesday { get; set; }
        [NinjaScriptProperty]
        [Display(Name = "Trade Wednesday", Order = 4, GroupName = "Weekday Filter")] public bool TradeWednesday { get; set; }
        [NinjaScriptProperty]
        [Display(Name = "Trade Thursday", Order = 5, GroupName = "Weekday Filter")] public bool TradeThursday { get; set; }
        [NinjaScriptProperty]
        [Display(Name = "Trade Friday", Order = 6, GroupName = "Weekday Filter")] public bool TradeFriday { get; set; }
        [NinjaScriptProperty]
        [Display(Name = "Trade Saturday", Order = 7, GroupName = "Weekday Filter")] public bool TradeSaturday { get; set; }
        [NinjaScriptProperty]
        [Display(Name = "Trade Sunday", Order = 8, GroupName = "Weekday Filter")] public bool TradeSunday { get; set; }
        #endregion

        #region Month Filter
        [NinjaScriptProperty]
        [Display(Name = "Use Month Filter", Order = 1, GroupName = "Month Filter")]
        public bool UseMonthFilter { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "January", Order = 2, GroupName = "Month Filter")] public bool TradeJanuary { get; set; }
        [NinjaScriptProperty]
        [Display(Name = "February", Order = 3, GroupName = "Month Filter")] public bool TradeFebruary { get; set; }
        [NinjaScriptProperty]
        [Display(Name = "March", Order = 4, GroupName = "Month Filter")] public bool TradeMarch { get; set; }
        [NinjaScriptProperty]
        [Display(Name = "April", Order = 5, GroupName = "Month Filter")] public bool TradeApril { get; set; }
        [NinjaScriptProperty]
        [Display(Name = "May", Order = 6, GroupName = "Month Filter")] public bool TradeMay { get; set; }
        [NinjaScriptProperty]
        [Display(Name = "June", Order = 7, GroupName = "Month Filter")] public bool TradeJune { get; set; }
        [NinjaScriptProperty]
        [Display(Name = "July", Order = 8, GroupName = "Month Filter")] public bool TradeJuly { get; set; }
        [NinjaScriptProperty]
        [Display(Name = "August", Order = 9, GroupName = "Month Filter")] public bool TradeAugust { get; set; }
        [NinjaScriptProperty]
        [Display(Name = "September", Order = 10, GroupName = "Month Filter")] public bool TradeSeptember { get; set; }
        [NinjaScriptProperty]
        [Display(Name = "October", Order = 11, GroupName = "Month Filter")] public bool TradeOctober { get; set; }
        [NinjaScriptProperty]
        [Display(Name = "November", Order = 12, GroupName = "Month Filter")] public bool TradeNovember { get; set; }
        [NinjaScriptProperty]
        [Display(Name = "December", Order = 13, GroupName = "Month Filter")] public bool TradeDecember { get; set; }
        #endregion

		private bool _isRealtime = false;

		private bool   _wasInPosition   = false;
        private double _initialEntryRef = 0.0;
        private double _initialRisk     = 0.0;
        private bool   _initialSet      = false;

        private double _signalHigh = 0.0;
        private double _signalLow = 0.0;

        private bool _setupArmed = false;
        private bool _submissionPending = false;
        private bool _breakoutTriggered = false;

		private const string STOP_LIMIT_EXIT_NAME = "ProtectiveStopLimit";
		private Order _stopLimitExitOrder = null;


        private Order _limitOrder = null;

        private const string LIMIT_ORDER_NAME = "BuyLimitEntry";

        private HashSet<int> _allowedHoursSet = new HashSet<int>();

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "Red-Green Breakout EA: arms on red candle, submits one Buy Limit when price breaks above candle high.";
                Name = "RedGreenBreakoutEA";
                Calculate = Calculate.OnEachTick;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.UniqueEntries;
				StartBehavior = StartBehavior.ImmediatelySubmit;
                IsExitOnSessionCloseStrategy = false;

                Quantity = 1;
                RiskReward = 1.0;
                LimitOffsetPercent = 25.0;

                UseCandleRangeFilter = false;
                MaxCandleRange = 50.0;
                MinCandleRange = 5.0;

                UseFlattenEnd = true;
                FlattenHourEnd = 20;
                FlattenMinuteEnd = 0;

                UseTradeWindow = true;
                AllowedHours = "10,11,12,13,14,15,16,17,18,19";

                UseWeekdayFilter = false;
                TradeMonday = true;
                TradeTuesday = true;
                TradeWednesday = true;
                TradeThursday = true;
                TradeFriday = true;
                TradeSaturday = true;
                TradeSunday = true;

                UseMonthFilter = false;
                TradeJanuary = true;
                TradeFebruary = true;
                TradeMarch = true;
                TradeApril = true;
                TradeMay = true;
                TradeJune = true;
                TradeJuly = true;
                TradeAugust = true;
                TradeSeptember = true;
                TradeOctober = true;
                TradeNovember = true;
                TradeDecember = true;
            }
			else if (State == State.Realtime)
			{
				_isRealtime = true;
				ResetSetup();
				Print("=== Realtime started: strategy state reset ===");
			}

            else if (State == State.Configure)
            {
                _allowedHoursSet.Clear();

                if (!string.IsNullOrWhiteSpace(AllowedHours))
                {
                    foreach (var part in AllowedHours.Split(','))
                    {
                        int h;
                        if (int.TryParse(part.Trim(), out h) && h >= 0 && h <= 23)
                            _allowedHoursSet.Add(h);
                    }
                }
            }
        }

		protected override void OnOrderUpdate(Order order,
									  double limitPrice,
									  double stopPrice,
									  int quantity,
									  int filled,
									  double averageFillPrice,
									  OrderState orderState,
									  DateTime time,
									  ErrorCode error,
									  string comment)
        {
            if (order == null || order.Name != LIMIT_ORDER_NAME)
                return;

            _limitOrder = order;

            switch (orderState)
            {
                case OrderState.Working:
                    _breakoutTriggered = true;
                    _submissionPending = false;
                    Print(string.Format("📋 BuyLimit Working @ {0:F5}", limitPrice));
                    break;

                case OrderState.PartFilled:
                    _breakoutTriggered = true;
                    _submissionPending = false;

                    if (!_initialSet)
                    {
                        _initialEntryRef = _signalHigh;
                        _initialRisk     = _signalHigh - _signalLow;
                        _initialSet      = _initialRisk > 0.0;
                    }

                    Print(string.Format("📌 BuyLimit PartFilled @ {0:F5}", averageFillPrice));
                    break;

                case OrderState.Filled:
                    _breakoutTriggered = true;
                    _submissionPending = false;

                    _initialEntryRef = _signalHigh;
                    _initialRisk     = _signalHigh - _signalLow;
                    _initialSet      = _initialRisk > 0.0;

                    Print(string.Format("✅ BuyLimit Filled @ {0:F5}", averageFillPrice));
                    break;

                case OrderState.Cancelled:
                case OrderState.Rejected:
                    _submissionPending = false;
                    _breakoutTriggered = false;
                    _limitOrder = null;
                    Print(string.Format("⚠️ BuyLimit {0} ({1}) - retry enabled", orderState, error));
                    break;
            }
        }




        protected override void OnBarUpdate()
        {
            if (CurrentBar < 1)
                return;

            DateTime barTime = Time[0];

            if (UseFlattenEnd &&
                barTime.Hour == FlattenHourEnd &&
                barTime.Minute == FlattenMinuteEnd)
            {
                FlattenAll("Flatten end-of-session");
                return;
            }

            if (_isRealtime
				&& _setupArmed
				&& !_breakoutTriggered
				&& !_submissionPending
				&& Position.MarketPosition == MarketPosition.Flat
				)

            {
                double ask = GetCurrentAsk();
                if (ask >= _signalHigh)
                {
                    Print(string.Format("⚡ Breakout @ Ask={0:F5} - submitting BuyLimit", ask));
                    SubmitBuyLimit();
                }
            }

            if (BarsInProgress != 0 || !IsFirstTickOfBar)
                return;

            bool isInPosition = Position.MarketPosition == MarketPosition.Long;

            if (!isInPosition && _wasInPosition)
			{
				_initialEntryRef = 0.0;
				_initialRisk     = 0.0;
				_initialSet      = false;
				_limitOrder      = null;
				_stopLimitExitOrder = null;
				_setupArmed      = false;
				_submissionPending = false;
				_breakoutTriggered = false;
			}


            _wasInPosition = isInPosition;

            if (UseWeekdayFilter && !IsWeekdayAllowed(barTime))
            {
                Print("📅 Weekday blocked: " + barTime.DayOfWeek);
                return;
            }

            if (UseMonthFilter && !IsMonthAllowed(barTime))
            {
                Print("📅 Month blocked: " + barTime.ToString("MMMM"));
                return;
            }

            if (Position.MarketPosition != MarketPosition.Flat)
            {
                if (_initialSet && _initialRisk > 0.0)
                {
                    double target = _initialEntryRef + _initialRisk * RiskReward;
                    double priorBarClose = Close[1];

                    if (priorBarClose >= target) 	// Bar closed at or above target
                    {
                        Print(string.Format("🎯 Bar-close target reached | Close[1]={0:F5} Target={1:F5}", priorBarClose, target));
                        ExitLong("RRClose", LIMIT_ORDER_NAME);
                    }
                }

                return;
            }

            if (UseTradeWindow && !IsTimeWindowAllowed(barTime))
            {
                Print("⏱ Outside trading window: " + barTime.Hour + ":00");
                return;
            }

            double o1 = Open[1];
            double h1 = High[1];
            double l1 = Low[1];
            double c1 = Close[1];

            if (c1 < o1)
            {
                double range = h1 - l1;
                if (range <= 0.0)
                    return;

                if (UseCandleRangeFilter)
                {
                    double rangeTicks = range / TickSize;

                    if (rangeTicks > MaxCandleRange)
                    {
                        Print(string.Format("⚠️ Range {0:F1} ticks > Max {1:F1} - skipped", rangeTicks, MaxCandleRange));
                        return;
                    }

                    if (rangeTicks < MinCandleRange)
                    {
                        Print(string.Format("⚠️ Range {0:F1} ticks < Min {1:F1} - skipped", rangeTicks, MinCandleRange));
                        return;
                    }
                }

                CancelPendingLimit();

                _signalHigh = h1;
                _signalLow = l1;
                _setupArmed = true;
                _submissionPending = false;
                _breakoutTriggered = false;

                Print(string.Format("🔴 Red candle armed | High={0:F5} Low={1:F5} Range={2:F5}",
                    h1, l1, range));
            }
        }


        private void SubmitBuyLimit()
        {
            if (_signalHigh <= 0.0 || _signalLow <= 0.0)
            {
                Print("⚠️ Invalid signal levels - submission skipped");
                return;
            }

            double candleRange = _signalHigh - _signalLow;
            if (candleRange <= 0.0)
            {
                Print("⚠️ Candle range = 0 - submission skipped");
                return;
            }

            double pct = Math.Max(0.0, Math.Min(100.0, LimitOffsetPercent));
            double entryPrice = _signalHigh - candleRange * (pct / 100.0);
            entryPrice = Instrument.MasterInstrument.RoundToTickSize(entryPrice);

            _submissionPending = true;
            EnterLongLimit(Quantity, entryPrice, LIMIT_ORDER_NAME);

            Print(string.Format("📤 BuyLimit submitted | Offset={0}% Entry={1:F5} RR={2}",
			pct, entryPrice, RiskReward));

        }

		protected override void OnExecutionUpdate(Execution execution,
                                          string executionId,
                                          double price,
                                          int quantity,
                                          MarketPosition marketPosition,
                                          string orderId,
                                          DateTime time)
		{
			if (!_isRealtime || execution == null || execution.Order == null)
				return;

			if (execution.Order.Name != LIMIT_ORDER_NAME)
				return;

			if (marketPosition != MarketPosition.Long)
				return;

			if (!_initialSet)
			{
				_initialEntryRef = _signalHigh;
				_initialRisk     = _signalHigh - _signalLow;
				_initialSet      = _initialRisk > 0.0;
			}

			if (_stopLimitExitOrder == null
				|| _stopLimitExitOrder.OrderState == OrderState.Cancelled
				|| _stopLimitExitOrder.OrderState == OrderState.Rejected
				|| _stopLimitExitOrder.OrderState == OrderState.Filled)
			{
				double stopPx  = Instrument.MasterInstrument.RoundToTickSize(_signalLow);
				double limitPx = Instrument.MasterInstrument.RoundToTickSize(stopPx - TickSize);

				ExitLongStopLimit(
					0,
					true,
					execution.Quantity,
					limitPx,
					stopPx,
					STOP_LIMIT_EXIT_NAME,
					LIMIT_ORDER_NAME
				);

				Print(string.Format("🛡 StopLimit submitted | Stop={0:F5} Limit={1:F5} Qty={2}",
					stopPx, limitPx, execution.Quantity));
			}
		}


        private void CancelPendingLimit()
        {
            if (_limitOrder != null
                && _limitOrder.OrderState != OrderState.Filled
                && _limitOrder.OrderState != OrderState.Cancelled
                && _limitOrder.OrderState != OrderState.Rejected)
            {
                CancelOrder(_limitOrder);
                Print("🧹 Previous BuyLimit cancelled");
            }

            _limitOrder = null;
            _submissionPending = false;
            _breakoutTriggered = false;

			if (_stopLimitExitOrder != null
				&& _stopLimitExitOrder.OrderState != OrderState.Filled
				&& _stopLimitExitOrder.OrderState != OrderState.Cancelled
				&& _stopLimitExitOrder.OrderState != OrderState.Rejected)
			{
				CancelOrder(_stopLimitExitOrder);
				Print("🧹 Previous StopLimit cancelled");
			}
			_stopLimitExitOrder = null;

        }

        private void FlattenAll(string reason)
        {
            if (Position.MarketPosition != MarketPosition.Flat)
            {
                ExitLong();
                Print("🌙 " + reason + " -> position closed");
            }

            CancelPendingLimit();
            ResetSetup();
            Print("🌙 " + reason + " -> done");
        }

        private void ResetSetup()
        {
            _signalHigh = 0.0;
            _signalLow = 0.0;
            _setupArmed = false;
            _submissionPending = false;
            _breakoutTriggered = false;
            _limitOrder = null;
			_stopLimitExitOrder = null;

            _initialEntryRef = 0.0;
            _initialRisk = 0.0;
            _initialSet = false;
            _wasInPosition = false;
        }


        private bool IsTimeWindowAllowed(DateTime t)
        {
            if (!UseTradeWindow)
                return true;

            return _allowedHoursSet.Contains(t.Hour);
        }

        private bool IsWeekdayAllowed(DateTime t)
        {
            if (!UseWeekdayFilter)
                return true;

            switch (t.DayOfWeek)
            {
                case DayOfWeek.Monday: return TradeMonday;
                case DayOfWeek.Tuesday: return TradeTuesday;
                case DayOfWeek.Wednesday: return TradeWednesday;
                case DayOfWeek.Thursday: return TradeThursday;
                case DayOfWeek.Friday: return TradeFriday;
                case DayOfWeek.Saturday: return TradeSaturday;
                case DayOfWeek.Sunday: return TradeSunday;
                default: return false;
            }
        }

        private bool IsMonthAllowed(DateTime t)
        {
            if (!UseMonthFilter)
                return true;

            switch (t.Month)
            {
                case 1: return TradeJanuary;
                case 2: return TradeFebruary;
                case 3: return TradeMarch;
                case 4: return TradeApril;
                case 5: return TradeMay;
                case 6: return TradeJune;
                case 7: return TradeJuly;
                case 8: return TradeAugust;
                case 9: return TradeSeptember;
                case 10: return TradeOctober;
                case 11: return TradeNovember;
                case 12: return TradeDecember;
                default: return false;
            }
        }
    }
}
