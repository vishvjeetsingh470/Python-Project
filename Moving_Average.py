for stock in stocks:
    plt.figure(figsize=(12,5))
    plt.plot(data[stock], label=f"{stock} Price")
    plt.plot(ma_data[f"{stock}_MA50"], label=f"{stock} MA50")
    plt.plot(ma_data[f"{stock}_MA200"], label=f"{stock} MA200")

    plt.title(f"{stock} - 50 & 200 Day Moving Average")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()
