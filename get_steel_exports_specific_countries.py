import os
import pandas as pd
from dotenv import load_dotenv
import comtradeapicall

load_dotenv()

PRIMARY_KEY = os.getenv('COMTRADE_PRIMARY_KEY')

COUNTRY_CODES = {
    'Belgium': '56',
    'Italy': '380',
    'United Arab Emirates': '784'
}

def get_steel_exports_to_countries(years=10):
    """
    Get Chinese steel exports (HS code 72) to Belgium, Italy, and UAE
    for the past specified number of years
    
    Parameters:
    - Reporter: China (code 156)
    - HS Code: 72 (Iron and steel)
    - Flow: Exports (X)
    - Partners: Belgium (56), Italy (380), UAE (784)
    - Period: Last 10 years
    """
    
    if not PRIMARY_KEY:
        print("❌ API key not found. Please set COMTRADE_PRIMARY_KEY in .env file")
        return None
    
    current_year = 2023
    start_year = current_year - years + 1
    
    print("Fetching Chinese steel export data (HS Code 72)")
    print(f"Countries: Belgium, Italy, United Arab Emirates")
    print(f"Period: {start_year}-{current_year}")
    print("=" * 80)
    
    all_data = []
    
    try:
        # Test with simple parameters first - one year at a time
        for year in range(start_year, current_year + 1):
            print(f"\nFetching data for year {year}...")
            
            for country_name, country_code in COUNTRY_CODES.items():
                print(f"  - {country_name} (code: {country_code})...", end=" ")
                
                try:
                    result = comtradeapicall.getFinalData(
                        subscription_key=PRIMARY_KEY,
                        typeCode='C',
                        freqCode='A',
                        clCode='HS',
                        period=str(year),
                        reporterCode='156',
                        cmdCode='72',
                        flowCode='X',
                        partnerCode=country_code,
                        partner2Code=None,
                        customsCode=None,
                        motCode=None,
                        maxRecords=100,
                        format_output='JSON',
                        aggregateBy=None,
                        breakdownMode='classic',
                        countOnly=None,
                        includeDesc=True
                    )
                    
                    if result is not None:
                        temp_df = pd.DataFrame(result) if not isinstance(result, pd.DataFrame) else result
                        if not temp_df.empty:
                            all_data.append(temp_df)
                            print(f"✅ {len(temp_df)} records")
                        else:
                            print("⚠️ No data")
                    else:
                        print("⚠️ No data")
                        
                except Exception as e:
                    print(f"❌ Error: {str(e)}")
        
        if not all_data:
            print("\n❌ No data found for any year/country combination")
            return None
        
        result = pd.concat(all_data, ignore_index=True)
        
        df = result
        
        print(f"\n✅ Total records retrieved: {len(df)}")
        print(f"Columns available: {', '.join(df.columns.tolist())}\n")
        
        if 'primaryValue' in df.columns and 'partnerDesc' in df.columns and 'period' in df.columns:
            df_sorted = df.sort_values(['partnerDesc', 'period'])
            
            print("=" * 100)
            print("CHINESE STEEL EXPORTS (HS 72) TO BELGIUM, ITALY & UAE")
            print("=" * 100)
            
            for country in ['Belgium', 'Italy', 'United Arab Emirates']:
                country_data = df_sorted[df_sorted['partnerDesc'] == country]
                
                if not country_data.empty:
                    print(f"\n{'─' * 100}")
                    print(f"📍 {country.upper()}")
                    print(f"{'─' * 100}")
                    print(f"{'Year':<10} {'Export Value (USD)':<25} {'Quantity':<20} {'Net Weight (kg)':<20}")
                    print("─" * 100)
                    
                    for _, row in country_data.iterrows():
                        year = row.get('period', 'N/A')
                        value = row.get('primaryValue', 0)
                        qty = row.get('qty', 'N/A')
                        net_weight = row.get('netWgt', 'N/A')
                        
                        qty_str = f"{qty:,.0f}" if isinstance(qty, (int, float)) else str(qty)
                        net_wgt_str = f"{net_weight:,.0f}" if isinstance(net_weight, (int, float)) else str(net_weight)
                        
                        print(f"{year:<10} ${value:>22,.2f}  {qty_str:<20} {net_wgt_str:<20}")
                    
                    total_value = country_data['primaryValue'].sum()
                    print("─" * 100)
                    print(f"{'TOTAL':<10} ${total_value:>22,.2f}")
                else:
                    print(f"\n⚠️ No data found for {country}")
            
            print("\n" + "=" * 100)
            
            summary_df = df_sorted.groupby('partnerDesc').agg({
                'primaryValue': 'sum',
                'qty': 'sum',
                'netWgt': 'sum'
            }).reset_index()
            summary_df.columns = ['Country', 'Total Export Value (USD)', 'Total Quantity', 'Total Net Weight (kg)']
            summary_df = summary_df.sort_values('Total Export Value (USD)', ascending=False)
            
            print("\n📊 SUMMARY BY COUNTRY")
            print("=" * 100)
            print(summary_df.to_string(index=False))
            print("=" * 100)
            
            output_file = 'chinese_steel_exports_specific_countries.csv'
            df_sorted.to_csv(output_file, index=False)
            print(f"\n✅ Detailed data saved to: {output_file}")
            
            summary_file = 'chinese_steel_exports_summary.csv'
            summary_df.to_csv(summary_file, index=False)
            print(f"✅ Summary data saved to: {summary_file}")
            
            return df_sorted
        else:
            print("⚠️ Expected columns not found in the response")
            print(f"Available columns: {df.columns.tolist()}")
            
            output_file = 'chinese_steel_exports_raw_data.csv'
            df.to_csv(output_file, index=False)
            print(f"\n✅ Raw data saved to: {output_file}")
            return df
            
    except Exception as e:
        print(f"❌ Error fetching data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("\n" + "🔍 " * 40)
    print("CHINESE STEEL EXPORTS ANALYSIS - SPECIFIC COUNTRIES")
    print("🔍 " * 40 + "\n")
    
    result = get_steel_exports_to_countries(years=10)
    
    if result is not None:
        print("\n✅ Successfully retrieved export data for Belgium, Italy, and UAE!")
    else:
        print("\n❌ Failed to retrieve data. Please check your API key and parameters.")
