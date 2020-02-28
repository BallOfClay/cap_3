# Basic Info
Database Page: https://www.kaggle.com/msainani/gsmarena-mobile-devices 
Website: https://www.gsmarena.com/

# Data Pipeline Tasks:
    * Date Announced Conversion !
    * Is Tablet or Phone or Watch !
        * 7 in screen or bigger, https://www.cnet.com/news/what-makes-a-tablet-a-tablet-faq/)
            * Contains: Tab, Tablet, pad
        * Contains: watch, band, gear, fit, 
            * has screen smaller than 2" after 2016
    * Is major competitor (based off year?)
    * Format: 
        misc_price !
        body_weight 
        display_size 
        body_dimensions 
        display_resolution 
        battery 
        memory_internal 
        cameras:
            * main_camera_single
            * main_camera_video !
            * selfie_camera_single
            selfie_camera_video
            main_camera_features
            selfie_camera_features
            tests_camera
            selfie_camera
            * main_camera_dual
            selfie_camera_dual
            camera
            * main_camera_triple
            selfie_camera_v1
            main_camera
            * main_camera_quad
            selfie_camera_triple
            main_camera_v1
            main_camera_dual_or_triple
            main_camera_five
    * Encode features_sensors: accelerometer, gyro, heart rate, fingerprint, compass, proximity, barometer, SpO2, iris scanner, gesture, UV, tempurature, altimeter, infrared face recognition, face id !!!
    * Encode display_type: LCD, OLED, AMOLED, TFT 
    * Encode battery_charging: wireless charging, fast battery charging 
    * Encode body: water resistant, water proof, Pay, Stylus 
    * Encode cameras: wide, ultrawide, telephoto, panorama 
    * Encode main camera: LED, HDR 
    * Encode network_technology 
    * Encode wlan: a, b, g, n, ac, ax 
    * Encode sound_loudspeaker 

    * Parse out CPU speed 
    * Parse out pixel density and/or screen size 
    * Parse out battery capacities 
    * Parse out each antenna band 
    * Parse out network speed 
    * Encode body/back characteristics 

    Attributes per Feature:
    * Does Company Release Feature 
    * Does Apple Ever Release Feature 
    * When is feature first released 
    * When is feature first released by all companies 
    * When is feature released by Apple 
    * Is Apple First 
    * Length of time spefic between company release and first company release 

# Data Addition Tasks:
    * Is Flagship 
    * Is Budgetary Phone 
    * Company Country of Origin 
    * Device Target Market 

    * Cluster by year without oem name to see if I get anything interesting 

    * Understand GSMArena Python Webscrape 
    * Execute Webscrape for Ratings, Saves and Hits, Reviews, Discussions 
    * Execute Python Webscrape for 2019-Present 
    * Determine which countries/regions manufacture focuses on 
    * When each Network/Protocal was adopted 
    * Which bands available each region and when 

# Functions & Classes:
    Cleaning:
        * Drop What? (no cellular connectivity, insignificant manufactures)
        * Encode function based off of splits

    * FEATURE ANALYSIS CLASS:
        * is eventual Apple feature
        * is eventual Company feature
        * is first manufacturer
        * distance of company from first adoption company
        * Find product families via Levenshtein Distance
        * Anova Test
        * Set Timeframe
            * Aggregate per year/6mon before & after Apple
            * Year over year trend

    Plots:
        (see Data Products)
    
# EDA Tasks:
    * Number of handsets released each year, theme: consolidation
    * When phones are released in year
    * When new frequency bands are available vs network adoption
    * Progression of megapixels cameras
    * Progression of display size
    * Progression of screen pixels/resolution
    * Progression of battery capacity vs phone weight
    * Progression of CPU Speed
        * Does CPU performance correlate to Moore's Law (may have to find way to equate speeds and number of transistors)

# Further Research Tasks:
    * 

# Unsupervised Clustering:
    * perform random clusters on phones for given year 
        (find target audience,  )

# Feature Engineering:
    * 

# Relavent Features:


# Statistical Tests:
    * ANOVA, t-test

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


# Scripts:
    * Clean .py
    * Feature class .py
    * EDA .py

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