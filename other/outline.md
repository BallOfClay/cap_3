# Basic Info
Database Page: https://www.kaggle.com/msainani/gsmarena-mobile-devices 
Website: https://www.gsmarena.com/

# Data Pipeline Tasks:
    * Date Announced Conversion
    * Is Tablet or Phone or Watch 
        * 7 in screen or bigger, https://www.cnet.com/news/what-makes-a-tablet-a-tablet-faq/)
            * Contains: Tab, Tablet, pad
        * Contains: watch, band, gear, fit, 
            * has screen smaller than 2" after 2016
    * Is major competitor (based off year?)
    * Encode network technology
    * Encode water resistant, water proof, Pay, Stylus
    * Encode Display Type: LCD, OLED, AMOLED, TFT
    * Encode sound_loudspeaker
    * Encode wlan: a, b, g, n, ac, ax
    * Encode sensors: accelerometer, gyro, heart rate, fingerprint, compass, proximity, barometer, SpO2, iris scanner, gesture, UV, tempurature, altimeter
    * Encode main camera: LED, HDR
    * Encode wireless charging, fast battery charging
    * Encode cameras: wide, ultrawide, telephoto,
    * Parse Prices

    * Parse out CPU speed
    * Parse out pixel density and/or screen size
    * Parse out battery capacities
    * Parse out each antenna band
    * Parse out network speed
    * Encode body/back characteristics

# Data Addition Tasks:
    * Is Flagship
    * Is Budgetary Phone
    * Company Country of Origin
    * Device Target Market

    * Understand GSMArena Python Webscrape
    * Execute Webscrape for Ratings, Saves and Hits, Reviews, Discussions
    * Execute Python Webscrape for 2019-Present
    * Determine which countries/regions manufacture focuses on
    * When each Network/Protocal was adopted
    * Which bands available each region and when

# Functions & Classes:
    * Drop What? (no cellular connectivity, insignificant manufactures)
    * Aggregate per year/6mon before & after Apple
    * Year over year trend
    * Anova Test
    * Encode function based off of
    * is eventual Apple feature
    * is first manufacturer
    * distance from first adoption


# EDA Tasks:
    * Number of handsets released each year, theme: consolidation
    * When phones are released in year
    * When new frequency bands are available vs network adoption
    * Progression of megapixels cameras
    * Progression of display size
    * Progression of pixels
    * Progression of battery capacity vs phone weight
    * Does CPU performance correlate to Moore's Law (may have to find way to equate speeds and number of transistors)

# Further Research Tasks:
    * 

# Unsupervised Clustering:


# Feature Engineering:
    * 

# Relavent Features:


# Statistical Tests:
    *

# Supervised Modeling:
    * KNN

# Data Products:
    * Apple attribute uptake compared to competition
    * Screen size growth over time
    * Battery capicity over time
    * Battery capicity versus weight over time
    * Audio Jack Removal
    * Battery Capacity vs Weight
    * Features per Dollar
    * Uptake Comparison

# README Tasks:


# Things Completed:


# Next Steps:
    * 

# Questions:
    * 

# Chosen Models
    * 

# Feature Selection Models:
    ExtraTreesClassifier
    * GradientBoostingClassifier
    * RandomForestClassifier
    XGBClassifier
    RFE
    SelectPercentile
    * PCA
    PCA + SelectPercentile
    Feature Engineering

# Evaluation and Prediction Models:
    LDA (Linear algo)
    * LR (Linear algo)
    * KNN (Non-linear algo)
    CART (Non-linear algo)      ******************
    * Naive Bayes (Non-linear algo)
    SVC (Non-linear algo)
    * Bagged Decision Trees (Bagging)
    * Random Forest (Bagging)
    Extra Trees (Bagging)
    AdaBoost (Boosting)
    Stochastic Gradient Boosting (Boosting)
    Voting Classifier (Voting)
    MLP (Deep Learning)
    * XGBoost


# Parameter Analysis:
Capstone 1:
    * Attributes: brand	 model	network_technology	2G_bands	3G_bands	4G_bands	network_speed	GPRS	EDGE	announced	status	dimentions	weight_g	weight_oz	SIM	display_type	display_resolution	display_size	OS	CPU	Chipset	GPU	memory_card	internal_memory	RAM	primary_camera	secondary_camera	loud_speaker	audio_jack	WLAN	bluetooth	GPS	NFC	radio	USB	sensors	battery	colors	approx_price_EUR	img_url	
    * Numerical Data: 2G_bands  3G_bands    4G_bands    network_speed   dimensions  weight_g    weight_oz   display_resolution  display_size    CPU mememory_card   internal_memory RAM primary_camera  secondary_camera    battery approx_price_EUR
    * Additional Data: 5G   SIM_multiple    form_factor keyboard    notch OIS   telephoto   ultrawide   pop-up_camera   dual_speakers   barometer   heart_rate  fingerprint infrared    removable   charging_speed  wireless_charging   popularity
Capstone 3:


# Relavent Features:
Capstone 1:
    * Sensors
    ** Accelerometer
    ** proximity
    ** compass
    ** gyro
    ** barrometer
    ** Fingerprint
    ** Iris Scanner
    ** heart rate ^
    ** humidity ^
    ** UV, UV light
    ** gesture ^
    ** tempurature ^
    ** altimeter ^
    ** SpO2 ^
    * network_technology
    * network_speed
    * Radio
    * GPS
    * CPU speed
    * NFC
    * Screen type
    * SIM type
    * Display type
    * Display size
    * Internal Memory size
    * RAM size
    * Bluetooth level
    * WiFi level
    * Battery size
    * Secondary camera

Capstone 3:
    * 