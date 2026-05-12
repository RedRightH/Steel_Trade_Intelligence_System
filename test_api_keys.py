import os
from dotenv import load_dotenv
import comtradeapicall

load_dotenv()

PRIMARY_KEY = os.getenv('COMTRADE_PRIMARY_KEY')
SECONDARY_KEY = os.getenv('COMTRADE_SECONDARY_KEY')

def test_primary_key():
    """Test the primary API key with a simple request"""
    print("Testing Primary Key...")
    print(f"Key: {PRIMARY_KEY[:10]}..." if PRIMARY_KEY else "No primary key found")
    
    if not PRIMARY_KEY:
        print("❌ Primary key not found in .env file")
        return False
    
    try:
        result = comtradeapicall.getFinalData(
            subscription_key=PRIMARY_KEY,
            typeCode='C',
            freqCode='A',
            clCode='HS',
            period='2023',
            reporterCode='356',
            cmdCode='TOTAL',
            flowCode='X',
            partnerCode='0',
            partner2Code=None,
            customsCode=None,
            motCode=None,
            maxRecords=10,
            format_output='JSON',
            aggregateBy=None,
            breakdownMode='classic',
            countOnly=None,
            includeDesc=True
        )
        
        print("✅ Primary key is valid!")
        print(f"Sample response received with {len(result)} records")
        return True
        
    except Exception as e:
        print(f"❌ Primary key test failed: {str(e)}")
        return False

def test_secondary_key():
    """Test the secondary API key with a simple request"""
    print("\nTesting Secondary Key...")
    print(f"Key: {SECONDARY_KEY[:10]}..." if SECONDARY_KEY else "No secondary key found")
    
    if not SECONDARY_KEY:
        print("❌ Secondary key not found in .env file")
        return False
    
    try:
        result = comtradeapicall.getFinalData(
            subscription_key=SECONDARY_KEY,
            typeCode='C',
            freqCode='A',
            clCode='HS',
            period='2023',
            reporterCode='356',
            cmdCode='TOTAL',
            flowCode='X',
            partnerCode='0',
            partner2Code=None,
            customsCode=None,
            motCode=None,
            maxRecords=10,
            format_output='JSON',
            aggregateBy=None,
            breakdownMode='classic',
            countOnly=None,
            includeDesc=True
        )
        
        print("✅ Secondary key is valid!")
        print(f"Sample response received with {len(result)} records")
        return True
        
    except Exception as e:
        print(f"❌ Secondary key test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("UN Comtrade API Key Testing")
    print("=" * 60)
    
    primary_valid = test_primary_key()
    secondary_valid = test_secondary_key()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"Primary Key: {'✅ Valid' if primary_valid else '❌ Invalid'}")
    print(f"Secondary Key: {'✅ Valid' if secondary_valid else '❌ Invalid'}")
    print("=" * 60)
