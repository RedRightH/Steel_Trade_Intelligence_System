import os
import pandas as pd
from dotenv import load_dotenv
import comtradeapicall

load_dotenv()

PRIMARY_KEY = os.getenv('COMTRADE_PRIMARY_KEY')

def get_top_10_steel_export_destinations():
    """
    Get top 10 export destinations for Indian steel exports (HS code 7208)
    
    Parameters:
    - Reporter: India (code 356)
    - HS Code: 7208 (Flat-rolled products of iron or non-alloy steel)
    - Flow: Exports (X)
    - Period: Most recent year (2023)
    """
    
    if not PRIMARY_KEY:
        print("❌ API key not found. Please set COMTRADE_PRIMARY_KEY in .env file")
        return None
    
    print("Fetching Indian steel export data (HS Code 7208)...")
    print("=" * 80)
    
    try:
        result = comtradeapicall.getFinalData(
            subscription_key=PRIMARY_KEY,
            typeCode='C',
            freqCode='A',
            clCode='HS',
            period='2023',
            reporterCode='356',
            cmdCode='7208',
            flowCode='X',
            partnerCode=None,
            partner2Code=None,
            customsCode=None,
            motCode=None,
            maxRecords=500,
            format_output='JSON',
            aggregateBy=None,
            breakdownMode='classic',
            countOnly=None,
            includeDesc=True
        )
        
        if result is None:
            print("No data found for the specified parameters")
            return None
        
        df = pd.DataFrame(result) if not isinstance(result, pd.DataFrame) else result
        
        if df.empty:
            print("No data found for the specified parameters")
            return None
        
        print(f"Total records retrieved: {len(df)}")
        print(f"\nColumns available: {', '.join(df.columns.tolist())}")
        
        if 'primaryValue' in df.columns and 'partnerDesc' in df.columns:
            df_sorted = df.sort_values('primaryValue', ascending=False)
            
            top_10 = df_sorted.head(10)
            
            print("\n" + "=" * 80)
            print("TOP 10 EXPORT DESTINATIONS FOR INDIAN STEEL (HS 7208) - 2023")
            print("=" * 80)
            print(f"\n{'Rank':<6} {'Country':<30} {'Export Value (USD)':<20} {'Quantity':<15}")
            print("-" * 80)
            
            for idx, row in enumerate(top_10.itertuples(), 1):
                country = getattr(row, 'partnerDesc', 'N/A')
                value = getattr(row, 'primaryValue', 0)
                qty = getattr(row, 'qty', 0) if hasattr(row, 'qty') else 'N/A'
                
                print(f"{idx:<6} {country:<30} ${value:>18,.2f} {str(qty):<15}")
            
            print("=" * 80)
            
            output_file = 'indian_steel_exports_top10.csv'
            top_10.to_csv(output_file, index=False)
            print(f"\n✅ Data saved to: {output_file}")
            
            return top_10
        else:
            print("⚠️ Expected columns not found in the response")
            print(f"Available columns: {df.columns.tolist()}")
            return df
            
    except Exception as e:
        print(f"❌ Error fetching data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = get_top_10_steel_export_destinations()
    
    if result is not None:
        print("\n✅ Successfully retrieved top 10 export destinations!")
    else:
        print("\n❌ Failed to retrieve data. Please check your API key and try again.")
