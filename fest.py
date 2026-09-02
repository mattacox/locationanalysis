# Festival Economic Impact Calculator (with fiscal offsets and walk-in spending)

def calc_economic_impact(
    total_attendees,
    local_population,
    local_attendance_rate,
    avg_spend_per_attendee,
    pct_local_vendors,
    local_vendor_multiplier=2.0,
    nonlocal_vendor_multiplier=1.2,
    public_costs=0.0,
    public_revenues=0.0,
    walk_in_spend_per_person=0.0,
    pct_attendees_who_shop_local=0.0,
    local_sales_tax_rate=0.02  # 2% local option sales tax
):
    """
    Estimate total *local economic impact* and *net fiscal impact*
    of a festival, including spillover retail spending and sales tax gains.

    Parameters
    ----------
    total_attendees : int
        Total number of attendees.
    local_population : int
        Number of people in the city or region.
    local_attendance_rate : float
        Fraction (0–1) of local population who attended.
    avg_spend_per_attendee : float
        Average spending per person at the festival ($).
    pct_local_vendors : float
        Percent (0–1) of vendors who are local.
    local_vendor_multiplier : float
        Multiplier for local vendor spending (default 2.0).
    nonlocal_vendor_multiplier : float
        Multiplier for non-local vendor spending (default 1.2).
    public_costs : float
        City’s total festival-related expenses (e.g. staff, security).
    public_revenues : float
        City’s revenues (vendor fees, sponsorships, etc.).
    walk_in_spend_per_person : float
        Average additional spend per attendee at local merchants ($).
    pct_attendees_who_shop_local : float
        Fraction (0–1) of all attendees who make additional local purchases.
    local_sales_tax_rate : float
        Local share of sales tax revenue (e.g. 0.02 for 2%).
    """

    # --- Separate local and visitor attendees ---
    local_attendees = local_population * local_attendance_rate
    visitor_attendees = max(total_attendees - local_attendees, 0)

    # Only visitor spending is new money
    direct_visitor_spending = visitor_attendees * avg_spend_per_attendee

    # Split spending between local and nonlocal vendors
    local_vendor_spending = direct_visitor_spending * pct_local_vendors
    nonlocal_vendor_spending = direct_visitor_spending * (1 - pct_local_vendors)

    # Apply multipliers
    local_vendor_impact = local_vendor_spending * local_vendor_multiplier
    nonlocal_vendor_impact = nonlocal_vendor_spending * nonlocal_vendor_multiplier

    # --- Walk-in merchant spillover ---
    total_walk_in_spend = total_attendees * pct_attendees_who_shop_local * walk_in_spend_per_person
    walk_in_impact = total_walk_in_spend * local_vendor_multiplier  # assume same multiplier

    # --- Sales tax revenue ---
    taxable_spending = local_vendor_spending + total_walk_in_spend
    local_sales_tax_revenue = taxable_spending * local_sales_tax_rate

    # --- Fiscal accounting ---
    total_public_revenues = public_revenues + local_sales_tax_revenue
    net_fiscal_impact = total_public_revenues - public_costs

    # --- Overall community impact ---
    total_local_impact = local_vendor_impact + nonlocal_vendor_impact + walk_in_impact
    net_community_benefit = total_local_impact + net_fiscal_impact

    return {
        "Local Attendees": round(local_attendees, 0),
        "Visitor Attendees": round(visitor_attendees, 0),
        "Direct Visitor Spending": round(direct_visitor_spending, 2),
        "Local Vendor Impact": round(local_vendor_impact, 2),
        "Nonlocal Vendor Impact": round(nonlocal_vendor_impact, 2),
        "Walk-in Local Spend": round(total_walk_in_spend, 2),
        "Walk-in Local Impact": round(walk_in_impact, 2),
        "Local Sales Tax Revenue": round(local_sales_tax_revenue, 2),
        "Public Revenues (incl. tax)": round(total_public_revenues, 2),
        "Public Costs": round(public_costs, 2),
        "Net Fiscal Impact": round(net_fiscal_impact, 2),
        "Total Local Impact": round(total_local_impact, 2),
        "Net Community Benefit": round(net_community_benefit, 2),
        "Overall Multiplier": round(total_local_impact / direct_visitor_spending, 2)
    }


# Example usage
result = calc_economic_impact(
    total_attendees=27600,
    local_population=9000,
    local_attendance_rate=0.75,
    avg_spend_per_attendee=70,
    pct_local_vendors=0.40,
    local_vendor_multiplier=2.0,
    nonlocal_vendor_multiplier=1.2,
    public_costs=135000,  # security, cleanup, admin
    public_revenues=135000,              # vendor + sponsorship revenue
    walk_in_spend_per_person=15,         # avg. $15 at local stores/restaurants
    pct_attendees_who_shop_local= .15,   # 25% of attendees also shop locally
    local_sales_tax_rate=0.02            # 2% local share
)

for k, v in result.items():
    print(f"{k}: {v:,}")
