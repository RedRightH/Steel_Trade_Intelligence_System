import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_steel_exports():
    """
    Plot Chinese steel exports to Belgium, Italy, and UAE over time
    """
    
    csv_file = 'chinese_steel_exports_specific_countries.csv'
    
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        print("Please run get_steel_exports_specific_countries.py first to generate the data.")
        return
    
    df = pd.read_csv(csv_file)
    
    print(f"✅ Loaded {len(df)} records from {csv_file}")
    print(f"Columns: {', '.join(df.columns.tolist())}\n")
    
    if 'period' not in df.columns or 'partnerDesc' not in df.columns or 'primaryValue' not in df.columns:
        print("❌ Required columns not found in the CSV file")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    df_pivot = df.pivot_table(
        index='period',
        columns='partnerDesc',
        values='primaryValue',
        aggfunc='sum'
    )
    
    plt.figure(figsize=(14, 8))
    
    colors = {
        'Belgium': '#1f77b4',
        'Italy': '#ff7f0e',
        'United Arab Emirates': '#2ca02c'
    }
    
    for country in df_pivot.columns:
        color = colors.get(country, None)
        plt.plot(
            df_pivot.index,
            df_pivot[country],
            marker='o',
            linewidth=2.5,
            markersize=8,
            label=country,
            color=color
        )
    
    plt.title('Chinese Steel Exports (HS 72) to Belgium, Italy & UAE\n2014-2023', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Year', fontsize=13, fontweight='bold')
    plt.ylabel('Export Value (USD)', fontsize=13, fontweight='bold')
    
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e9:.1f}B' if x >= 1e9 else f'${x/1e6:.0f}M'))
    
    plt.xticks(df_pivot.index, rotation=45)
    plt.tight_layout()
    
    output_file = 'chinese_steel_exports_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_file}")
    
    plt.show()
    
    print("\n📊 EXPORT VALUE SUMMARY:")
    print("=" * 80)
    for country in df_pivot.columns:
        total = df_pivot[country].sum()
        avg = df_pivot[country].mean()
        max_val = df_pivot[country].max()
        max_year = df_pivot[country].idxmax()
        
        print(f"\n{country}:")
        print(f"  Total (2014-2023): ${total:,.2f}")
        print(f"  Average per year:  ${avg:,.2f}")
        print(f"  Peak year:         {max_year} (${max_val:,.2f})")

if __name__ == "__main__":
    print("\n" + "📈 " * 40)
    print("CHINESE STEEL EXPORTS - VISUALIZATION")
    print("📈 " * 40 + "\n")
    
    plot_steel_exports()
