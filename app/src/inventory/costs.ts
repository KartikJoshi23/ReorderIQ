/**
 * Cost and constraint assumptions for the inventory policy (all adjustable on the dashboard).
 * Defaults are documented assumptions for a supermarket beverage line; capacity and budget scale
 * with the selected item's average demand so they make sense for any series.
 */
export interface CostParams {
  /** Purchase cost per unit ($). */
  unitCost: number;
  /** Annual holding cost as a share of unit cost (capital, space, handling). */
  holdingRatePerYear: number;
  /** Fixed cost per replenishment order ($): admin, transport, receiving. */
  orderCost: number;
  /** Cost per unit of demand lost to a stockout ($): lost margin + goodwill. */
  stockoutPenalty: number;
}

export const DEFAULT_COSTS: CostParams = { unitCost: 0.7, holdingRatePerYear: 0.25, orderCost: 120, stockoutPenalty: 0.5 };

export interface Constraints {
  /** Minimum fill rate (share of demand served from stock). */
  minFillRate: number;
  /** Storage capacity (units): the 95th-percentile peak on-hand must not exceed it. */
  capacity: number;
  /** Working-capital budget ($): average inventory value must not exceed it. */
  budget: number;
  /** Supplier minimum order quantity (units). */
  moq: number;
  /** Receiving-dock limit: maximum deliveries per week. */
  maxOrdersPerWeek: number;
}

/** Holding cost per unit per day ($). */
export function dailyHoldingCost(c: CostParams): number {
  return (c.unitCost * c.holdingRatePerYear) / 365;
}

/** Economic order quantity Q* = √(2·D·S / H) with D = annual demand, S = order cost, H = annual holding cost per unit. */
export function economicOrderQuantity(meanDailyDemand: number, c: CostParams): number {
  const annualDemand = meanDailyDemand * 365;
  const annualHolding = c.unitCost * c.holdingRatePerYear;
  return Math.sqrt((2 * annualDemand * c.orderCost) / annualHolding);
}

/** Defaults scaled to the item: capacity = 18 days of demand, budget = 12 days of demand at cost. */
export function defaultConstraints(meanDailyDemand: number, c: CostParams = DEFAULT_COSTS): Constraints {
  return {
    minFillRate: 0.98,
    capacity: Math.round(18 * meanDailyDemand),
    budget: Math.round(12 * meanDailyDemand * c.unitCost),
    moq: 0,
    maxOrdersPerWeek: 3,
  };
}
