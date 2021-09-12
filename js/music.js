.crx_bdwk_down_wrap {
    top: 70%;
    left: 0;
    position: fixed;
    z-index: 99999999;
    color: #fff;
    user-select: none;
}

    .crx_bdwk_down_wrap .crx_bdwk_down_loading {
        background-color: #666;
        cursor: wait;
        width: 126px;
        text-align: center;
        padding: 16px 0;
    }

    .crx_bdwk_down_wrap .crx_bdwk_down_loading p {
            font-size: 14px;
        }

    .crx_bdwk_down_wrap .crx_bdwk_down_loading small {
            font-size: 10px;
        }

    .crx_bdwk_down_wrap .crx_bdwk_down_btn {
        width: 126px;
        height: 60px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        font-size: 14px;
        background-color: #dd5a57;
        position: relative;
    }

    .crx_bdwk_down_wrap .crx_bdwk_down_types {
        display: flex;
        text-align: center;
        align-items: center;
        background-color: #666;
        font-size: 12px;
    }

    .crx_bdwk_down_wrap .crx_bdwk_down_types div {
            position: relative;
        }

    .crx_bdwk_down_wrap .crx_bdwk_down_types div:after {
                content: ' ';
                height: 12px;
                width: 1px;
                background-color: #eee;
                position: absolute;
                top: 10px;
                right: 0;
                transform: scaleX(0.5);
            }

    .crx_bdwk_down_wrap .crx_bdwk_down_types div:last-child:after {
                    display: none;
                }

    .crx_bdwk_down_wrap .crx_bdwk_down_types_check {
            flex: 1;
            color: #dd5a57;
            padding: 8px;
            cursor: pointer;
            font-weight: bold;
        }

    .crx_bdwk_down_wrap .crx_bdwk_down_types_uncheck {
            flex: 1;
            padding: 8px;
            cursor: pointer;
            color: #fff;
            font-weight: lighter;
        }
